#!/usr/bin/env python3
import argparse
from array import array
from pathlib import Path
import random

import ROOT


def make_file(path: Path, file_index: int, events: int, min_len: int, max_len: int, seed: int) -> None:
    rng = random.Random(seed + file_index)
    f = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("Events", "CUDA-Lab ROOT+CUDA synthetic input")

    x_true = ROOT.std.vector("float")()
    x_hat = ROOT.std.vector("float")()
    energy = ROOT.std.vector("float")()
    event_id = array("q", [0])

    tree.Branch("event_id", event_id, "event_id/L")
    tree.Branch("x_true", x_true)
    tree.Branch("x_hat", x_hat)
    tree.Branch("energy", energy)

    for ev in range(events):
        x_true.clear()
        x_hat.clear()
        energy.clear()
        event_id[0] = file_index * 10_000_000 + ev
        n = rng.randint(min_len, max_len)
        for _ in range(n):
            truth = rng.uniform(0.0, 2500.0)
            reco = truth + rng.gauss(0.0, 8.0)
            e = max(0.0, reco + rng.gauss(0.0, 2.0))
            x_true.push_back(truth)
            x_hat.push_back(reco)
            energy.push_back(e)
        tree.Fill()

    tree.Write()
    f.Close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--files", type=int, default=4)
    p.add_argument("--events", type=int, default=2000)
    p.add_argument("--min-len", type=int, default=32)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=9512)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.files):
        path = out / f"synthetic_{i:03d}.root"
        make_file(path, i, args.events, args.min_len, args.max_len, args.seed)
        print(path)


if __name__ == "__main__":
    main()
