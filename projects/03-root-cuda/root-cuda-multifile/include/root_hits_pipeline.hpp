#pragma once
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include "root_hits_gpu.cuh"
namespace fs=std::filesystem;

inline void cpu_reduce(const Batch& b,std::vector<double>& sums,std::vector<double>& maxs,std::vector<std::uint64_t>& counts,std::uint64_t& nonfinite){
    std::size_t n=b.offsets.size()-1; sums.assign(n,0); maxs.assign(n,0); counts.assign(n,0);
    for(std::size_t s=0;s<n;++s){ auto x0=b.offsets[s],x1=b.offsets[s+1]; double sum=0,mx=-std::numeric_limits<double>::infinity();
        for(auto i=x0;i<x1;++i){ double x=b.values[i]; if(!std::isfinite(x)){++nonfinite;continue;} sum+=x; mx=std::max(mx,x); }
        sums[s]=sum; maxs[s]=(x1>x0?mx:0); counts[s]=x1-x0;
    }
}
inline void accumulate(const SegmentMetric& m,Aggregate& a){ auto d=(std::size_t)m.detector; ++a.segments; if(m.pass)++a.segments_pass; a.hits_cpu[d]+=m.count_cpu; a.hits_gpu[d]+=m.count_gpu; a.sum_cpu[d]+=m.sum_cpu; a.sum_gpu[d]+=m.sum_gpu; a.max_cpu[d]=std::max(a.max_cpu[d],m.max_cpu); a.max_gpu[d]=std::max(a.max_gpu[d],m.max_gpu); }
inline void process_batch(const Batch& b,const Config& c,GpuWorkspace& g,Aggregate& a,FileSummary& f,TTree& tree,SegmentMetric& m){
    std::vector<double> cs,cm; std::vector<std::uint64_t> cc; std::uint64_t nf=0; auto t0=std::chrono::steady_clock::now(); cpu_reduce(b,cs,cm,cc,nf); auto t1=std::chrono::steady_clock::now(); auto gr=g.process(b.values,b.offsets);
    f.timing.cpu_compute_ms+=std::chrono::duration<double,std::milli>(t1-t0).count(); add_timing(f.timing,Timing{0,0,gr.timing.host_pack_ms,gr.timing.h2d_ms,gr.timing.kernel_ms,gr.timing.d2h_ms}); f.nonfinite_values+=nf; a.nonfinite_values+=nf;
    for(std::size_t s=0;s<cs.size();++s){ m.event_number=b.event_numbers[s];m.detector=b.detector_ids[s];m.count_cpu=cc[s];m.count_gpu=gr.counts[s];m.sum_cpu=cs[s];m.sum_gpu=gr.sums[s];m.max_cpu=cm[s];m.max_gpu=gr.maxs[s];m.pass=m.count_cpu==m.count_gpu&&close_enough(m.sum_cpu,m.sum_gpu,c)&&close_enough(m.max_cpu,m.max_gpu,c); ++f.segments;if(m.pass)++f.segments_pass;accumulate(m,a);tree.Fill(); }
}
inline FileSummary process_file(const std::string& p,const Config& c,GpuWorkspace& g,Aggregate& a,TTree& out,SegmentMetric& m){
    FileSummary f;f.path=p; std::unique_ptr<TFile> file(TFile::Open(p.c_str(),"READ")); if(!file||file->IsZombie()){f.message="cannot open ROOT file";return f;} TTree* tr=nullptr;file->GetObject(c.tree_name.c_str(),tr);if(!tr){f.message="missing tree";return f;}
    if(!tr->GetBranch("EventNumber")){f.message="missing EventNumber";return f;} for(auto b:kEnergyBranches)if(!tr->GetBranch(b)){f.message=std::string("missing ")+b;return f;}
    TTreeReader r(tr);TTreeReaderValue<Int_t> ev(r,"EventNumber");TTreeReaderArray<double> t(r,kEnergyBranches[0]),emb(r,kEnergyBranches[1]),emec(r,kEnergyBranches[2]),hec(r,kEnergyBranches[3]),fc(r,kEnergyBranches[4]);std::array<TTreeReaderArray<double>*,kDetectors> arr={&t,&emb,&emec,&hec,&fc};
    Batch b;std::size_t nev=0; while(true){auto q0=std::chrono::steady_clock::now();bool ok=r.Next();if(!ok)break;for(int d=0;d<kDetectors;++d){b.event_numbers.push_back(*ev);b.detector_ids.push_back(d);for(double x:*arr[d])b.values.push_back(x);b.offsets.push_back(b.values.size());}auto q1=std::chrono::steady_clock::now();f.timing.root_adapter_ms+=std::chrono::duration<double,std::milli>(q1-q0).count();++f.events;++nev;if(nev==c.batch_events){f.hits+=b.values.size();process_batch(b,c,g,a,f,out,m);b.clear();nev=0;}}
    if(nev){f.hits+=b.values.size();process_batch(b,c,g,a,f,out,m);} f.pass=f.events&&f.nonfinite_values==0&&f.segments==f.segments_pass;if(!f.pass&&f.message.empty())f.message="CPU/GPU gate failed or non-finite energy";++a.files;a.events+=f.events;add_timing(a.timing,f.timing);return f;
}
inline void write_csv(const Config& c,const std::vector<FileSummary>& files,const Aggregate& a,bool gate){fs::create_directories(c.output_dir);
    std::ofstream pf(fs::path(c.output_dir)/"per_file.csv");pf<<"file,pass,message,events,hits,segments,segments_pass,nonfinite_values,root_adapter_ms,cpu_compute_ms,host_pack_ms,h2d_ms,kernel_ms,d2h_ms\n"<<std::setprecision(12);for(auto&f:files)pf<<'"'<<f.path<<"\","<<(f.pass?"PASS":"FAIL")<<",\""<<f.message<<"\","<<f.events<<','<<f.hits<<','<<f.segments<<','<<f.segments_pass<<','<<f.nonfinite_values<<','<<f.timing.root_adapter_ms<<','<<f.timing.cpu_compute_ms<<','<<f.timing.host_pack_ms<<','<<f.timing.h2d_ms<<','<<f.timing.kernel_ms<<','<<f.timing.d2h_ms<<'\n';
    std::ofstream ps(fs::path(c.output_dir)/"per_subdetector.csv");ps<<"subdetector,hits_cpu,hits_gpu,energy_sum_cpu,energy_sum_gpu,energy_max_cpu,energy_max_gpu\n"<<std::setprecision(15);for(int d=0;d<kDetectors;++d)ps<<kDetectorNames[d]<<','<<a.hits_cpu[d]<<','<<a.hits_gpu[d]<<','<<(double)a.sum_cpu[d]<<','<<(double)a.sum_gpu[d]<<','<<a.max_cpu[d]<<','<<a.max_gpu[d]<<'\n';
    std::ofstream s(fs::path(c.output_dir)/"summary.csv");s<<"metric,value\n"<<std::setprecision(15)<<"files,"<<a.files<<"\nevents,"<<a.events<<"\nsegments,"<<a.segments<<"\nsegments_pass,"<<a.segments_pass<<"\nnonfinite_values,"<<a.nonfinite_values<<"\nroot_adapter_ms,"<<a.timing.root_adapter_ms<<"\ncpu_compute_ms,"<<a.timing.cpu_compute_ms<<"\nhost_pack_ms,"<<a.timing.host_pack_ms<<"\nh2d_ms,"<<a.timing.h2d_ms<<"\nkernel_ms,"<<a.timing.kernel_ms<<"\nd2h_ms,"<<a.timing.d2h_ms<<"\nresult_gate,"<<(gate?"PASS":"FAIL")<<'\n';}
