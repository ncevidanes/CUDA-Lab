#!/usr/bin/env bash
set -euo pipefail

REMOTE="${DRIVE_REMOTE:-gdrive-cuda:}"
REMOTE_DIR="${DRIVE_INPUT_DIR:-CUDA-Lab/root-input}"
SOURCE="${EXTERNAL_ROOT_DIR:-}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "RCLONE_GATE=FAIL reason=rclone_not_found" >&2
  exit 2
fi

if [[ -z "${SOURCE}" ]]; then
  mapfile -t roots < <(
    for base in "/media/${USER:-}" "/run/media/${USER:-}" /mnt; do
      [[ -d "$base" ]] || continue
      find "$base" -maxdepth 5 -type f -name '*.root' -printf '%h\n' 2>/dev/null || true
    done | sort | uniq -c | sort -nr
  )
  if [[ ${#roots[@]} -eq 0 ]]; then
    echo "EXTERNAL_ROOT_DISCOVERY_GATE=FAIL reason=no_root_files_found" >&2
    exit 3
  fi
  SOURCE="$(sed -E 's/^ *[0-9]+ //' <<<"${roots[0]}")"
fi

if [[ ! -d "$SOURCE" ]]; then
  echo "EXTERNAL_ROOT_DIR_GATE=FAIL path=$SOURCE" >&2
  exit 4
fi

ROOT_COUNT="$(find "$SOURCE" -type f -name '*.root' | wc -l)"
if [[ "$ROOT_COUNT" -eq 0 ]]; then
  echo "ROOT_FILE_COUNT_GATE=FAIL count=0 source=$SOURCE" >&2
  exit 5
fi

DEST="${REMOTE%:}:${REMOTE_DIR#/}"

echo "=================================================="
echo " CUDA LAB — EXTERNAL HD -> GOOGLE DRIVE STAGING"
echo "=================================================="
echo "SOURCE=$SOURCE"
echo "DEST=$DEST"
echo "ROOT_FILE_COUNT=$ROOT_COUNT"

rclone copy "$SOURCE" "$DEST" \
  --include '*.root' \
  --checksum \
  --transfers "${RCLONE_TRANSFERS:-4}" \
  --checkers "${RCLONE_CHECKERS:-8}" \
  --stats 30s \
  --stats-one-line

TMP_MANIFEST="$(mktemp)"
trap 'rm -f "$TMP_MANIFEST"' EXIT
rclone lsf "$DEST" --include '*.root' --recursive --files-only --format 'pst' > "$TMP_MANIFEST"
REMOTE_COUNT="$(wc -l < "$TMP_MANIFEST")"

if [[ "$REMOTE_COUNT" -lt "$ROOT_COUNT" ]]; then
  echo "DRIVE_STAGING_GATE=FAIL local=$ROOT_COUNT remote=$REMOTE_COUNT" >&2
  exit 6
fi

rclone copyto "$TMP_MANIFEST" "$DEST/_root_file_manifest.txt"

echo "REMOTE_ROOT_FILE_COUNT=$REMOTE_COUNT"
echo "DRIVE_STAGING_GATE=PASS"
