#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kDetectors = 5;
constexpr std::array<const char*, kDetectors> kDetectorNames = {
    "TileCal", "LArEMB", "LArEMEC", "LArHEC", "LArFCAL"};
constexpr std::array<const char*, kDetectors> kEnergyBranches = {
    "TileCalHit_energy", "LArHitEMB_energy", "LArHitEMEC_energy",
    "LArHitHEC_energy", "LArHitFCAL_energy"};

struct Config {
    std::string input_list, output_dir, tree_name = "CollectionTree";
    std::size_t batch_events = 512;
    int threads = 256;
    double rel_tol = 1e-10, abs_tol = 1e-9;
};
struct Timing {
    double root_adapter_ms=0, cpu_compute_ms=0, host_pack_ms=0;
    double h2d_ms=0, kernel_ms=0, d2h_ms=0;
};
struct SegmentMetric {
    std::int32_t event_number=0, detector=0;
    std::uint64_t count_cpu=0, count_gpu=0;
    double sum_cpu=0, sum_gpu=0, max_cpu=0, max_gpu=0;
    bool pass=false;
};
struct Aggregate {
    std::uint64_t files=0, events=0, segments=0, segments_pass=0, nonfinite_values=0;
    std::array<std::uint64_t,kDetectors> hits_cpu{}, hits_gpu{};
    std::array<long double,kDetectors> sum_cpu{}, sum_gpu{};
    std::array<double,kDetectors> max_cpu{}, max_gpu{};
    Timing timing;
};
struct FileSummary {
    std::string path, message;
    std::uint64_t events=0, hits=0, segments=0, segments_pass=0, nonfinite_values=0;
    Timing timing;
    bool pass=false;
};
struct Batch {
    std::vector<double> values;
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> event_numbers, detector_ids;
    void clear(){ values.clear(); offsets.assign(1,0); event_numbers.clear(); detector_ids.clear(); }
};

inline Config parse_args(int argc, char** argv){
    Config c;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto next=[&](const char* f){ if(i+1>=argc) throw std::runtime_error(std::string("missing value for ")+f); return std::string(argv[++i]); };
        if(a=="--input-list") c.input_list=next("--input-list");
        else if(a=="--output-dir") c.output_dir=next("--output-dir");
        else if(a=="--tree") c.tree_name=next("--tree");
        else if(a=="--batch-events") c.batch_events=std::stoull(next("--batch-events"));
        else if(a=="--threads") c.threads=std::stoi(next("--threads"));
        else if(a=="--rel-tol") c.rel_tol=std::stod(next("--rel-tol"));
        else if(a=="--abs-tol") c.abs_tol=std::stod(next("--abs-tol"));
        else if(a=="-h"||a=="--help"){ std::cout<<"--input-list FILE --output-dir DIR [--batch-events 512] [--threads 256]\n"; std::exit(0); }
        else throw std::runtime_error("unknown argument: "+a);
    }
    if(c.input_list.empty()||c.output_dir.empty()) throw std::runtime_error("--input-list and --output-dir are required");
    if(!c.batch_events) throw std::runtime_error("--batch-events must be > 0");
    return c;
}
inline std::vector<std::string> load_input_list(const std::string& p){
    std::ifstream in(p); if(!in) throw std::runtime_error("cannot open input list: "+p);
    std::vector<std::string> v; std::string s;
    while(std::getline(in,s)){ if(!s.empty()&&s.back()=='\r')s.pop_back(); auto f=s.find_first_not_of(" \t"); if(f==std::string::npos||s[f]=='#')continue; auto l=s.find_last_not_of(" \t"); v.push_back(s.substr(f,l-f+1)); }
    if(v.empty()) throw std::runtime_error("input list contains no files"); return v;
}
inline bool close_enough(double a,double b,const Config& c){
    if(!std::isfinite(a)||!std::isfinite(b)) return false;
    double s=std::max({1.0,std::abs(a),std::abs(b)}); return std::abs(a-b)<=c.abs_tol+c.rel_tol*s;
}
inline void add_timing(Timing& a,const Timing& b){
    a.root_adapter_ms+=b.root_adapter_ms; a.cpu_compute_ms+=b.cpu_compute_ms; a.host_pack_ms+=b.host_pack_ms;
    a.h2d_ms+=b.h2d_ms; a.kernel_ms+=b.kernel_ms; a.d2h_ms+=b.d2h_ms;
}
