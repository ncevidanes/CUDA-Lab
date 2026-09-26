#include <cuda_runtime.h>
#include <TFile.h>
#include <TNamed.h>
#include <TTree.h>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <utility>
#include "root_hits_pipeline.hpp"

int main(int argc,char** argv){try{
    Config c=parse_args(argc,argv);auto files=load_input_list(c.input_list);std::filesystem::create_directories(c.output_dir);
    int dev=0;CUDA_CHECK(cudaGetDevice(&dev));cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,dev));
    std::cout<<"==================================================\n CUDA LAB — ROOT HITS ENERGY CPU/GPU\n==================================================\n"<<"TREE="<<c.tree_name<<"\nINPUT_FILE_COUNT="<<files.size()<<"\nGPU_NAME="<<p.name<<"\nCUDA_CC="<<p.major<<'.'<<p.minor<<"\nBATCH_EVENTS="<<c.batch_events<<"\nTHREADS="<<c.threads<<'\n';
    GpuWorkspace gpu(c.threads);Aggregate a;a.max_cpu.fill(-std::numeric_limits<double>::infinity());a.max_gpu.fill(-std::numeric_limits<double>::infinity());std::vector<FileSummary> fs;
    TFile out((std::filesystem::path(c.output_dir)/"results.root").string().c_str(),"RECREATE");
    if(out.IsZombie()) throw std::runtime_error("cannot create results.root");
    TTree tree("EventDetectorMetrics","CPU/GPU metrics per event and calorimeter subsystem");tree.SetDirectory(&out);SegmentMetric m;
    tree.Branch("EventNumber",&m.event_number);tree.Branch("detector",&m.detector);tree.Branch("count_cpu",&m.count_cpu);tree.Branch("count_gpu",&m.count_gpu);tree.Branch("sum_cpu",&m.sum_cpu);tree.Branch("sum_gpu",&m.sum_gpu);tree.Branch("max_cpu",&m.max_cpu);tree.Branch("max_gpu",&m.max_gpu);tree.Branch("pass",&m.pass);
    for(std::size_t i=0;i<files.size();++i){std::cout<<"FILE_BEGIN="<<i+1<<'/'<<files.size()<<" PATH="<<files[i]<<'\n';auto f=process_file(files[i],c,gpu,a,tree,m);std::cout<<"FILE_GATE="<<(f.pass?"PASS":"FAIL")<<" EVENTS="<<f.events<<" HITS="<<f.hits<<" SEGMENTS="<<f.segments<<" SEGMENTS_PASS="<<f.segments_pass<<'\n';fs.push_back(std::move(f));}
    bool gate=a.events==100000&&a.segments==a.events*kDetectors&&a.segments_pass==a.segments&&a.nonfinite_values==0&&std::all_of(fs.begin(),fs.end(),[](auto&f){return f.pass;});
    out.cd();
    tree.Write("",TObject::kOverwrite);
    TNamed g("RESULT_GATE",gate?"PASS":"FAIL");g.Write("",TObject::kOverwrite);
    out.Write();
    out.Close();
    write_csv(c,fs,a,gate);
    double gpu_ms=a.timing.host_pack_ms+a.timing.h2d_ms+a.timing.kernel_ms+a.timing.d2h_ms;double sp=a.timing.kernel_ms?a.timing.cpu_compute_ms/a.timing.kernel_ms:0;double spi=gpu_ms?a.timing.cpu_compute_ms/gpu_ms:0;
    std::cout<<std::setprecision(12)<<"GLOBAL_FILES="<<a.files<<"\nGLOBAL_EVENTS="<<a.events<<"\nGLOBAL_SEGMENTS="<<a.segments<<"\nGLOBAL_SEGMENTS_PASS="<<a.segments_pass<<"\nNONFINITE_VALUES="<<a.nonfinite_values<<"\nROOT_ADAPTER_MS="<<a.timing.root_adapter_ms<<"\nCPU_COMPUTE_MS="<<a.timing.cpu_compute_ms<<"\nHOST_PACK_MS="<<a.timing.host_pack_ms<<"\nH2D_MS="<<a.timing.h2d_ms<<"\nKERNEL_MS="<<a.timing.kernel_ms<<"\nD2H_MS="<<a.timing.d2h_ms<<"\nCOMPUTE_SPEEDUP_CPU_OVER_KERNEL="<<sp<<"\nTRANSFER_INCLUSIVE_SPEEDUP="<<spi<<"\nRESULT_GATE="<<(gate?"PASS":"FAIL")<<'\n';return gate?0:2;
}catch(const std::exception&e){std::cerr<<"FATAL="<<e.what()<<'\n';return 1;}}
