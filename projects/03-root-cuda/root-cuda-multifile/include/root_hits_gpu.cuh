#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "root_hits_types.hpp"

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ std::ostringstream o; o<<"CUDA error "<<__FILE__<<':'<<__LINE__<<": "<<cudaGetErrorString(e); throw std::runtime_error(o.str()); } } while(0)

__global__ void segmented_reduce_kernel(const double* v,const std::uint64_t* off,double* sums,double* maxs,std::uint64_t* counts,std::size_t nseg){
    std::size_t s=blockIdx.x; if(s>=nseg)return;
    extern __shared__ unsigned char raw[]; double* ss=reinterpret_cast<double*>(raw); double* sm=ss+blockDim.x;
    unsigned t=threadIdx.x; auto b=off[s], e=off[s+1]; double sum=0, mx=-DBL_MAX;
    for(std::uint64_t i=b+t;i<e;i+=blockDim.x){ double x=v[i]; sum+=x; mx=fmax(mx,x); }
    ss[t]=sum; sm[t]=mx; __syncthreads();
    for(unsigned st=blockDim.x/2;st;st>>=1){ if(t<st){ ss[t]+=ss[t+st]; sm[t]=fmax(sm[t],sm[t+st]); } __syncthreads(); }
    if(t==0){ sums[s]=ss[0]; maxs[s]=(e>b?sm[0]:0.0); counts[s]=e-b; }
}

class GpuWorkspace {
public:
    struct Result { std::vector<double> sums,maxs; std::vector<std::uint64_t> counts; Timing timing; };
    explicit GpuWorkspace(int threads):threads_(threads){
        if(threads_<=0||threads_>1024||(threads_&(threads_-1))) throw std::runtime_error("threads must be power of two <=1024");
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&e0_)); CUDA_CHECK(cudaEventCreate(&e1_)); CUDA_CHECK(cudaEventCreate(&e2_)); CUDA_CHECK(cudaEventCreate(&e3_));
    }
    ~GpuWorkspace(){ release(); if(e3_)cudaEventDestroy(e3_); if(e2_)cudaEventDestroy(e2_); if(e1_)cudaEventDestroy(e1_); if(e0_)cudaEventDestroy(e0_); if(stream_)cudaStreamDestroy(stream_); }
    Result process(const std::vector<double>& values,const std::vector<std::uint64_t>& offsets){
        std::size_t seg=offsets.size()-1; ensure(values.size(),offsets.size(),seg);
        auto p0=std::chrono::steady_clock::now(); if(!values.empty())std::copy(values.begin(),values.end(),hv_); std::copy(offsets.begin(),offsets.end(),ho_); auto p1=std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(e0_,stream_));
        if(!values.empty())CUDA_CHECK(cudaMemcpyAsync(dv_,hv_,values.size()*sizeof(double),cudaMemcpyHostToDevice,stream_));
        CUDA_CHECK(cudaMemcpyAsync(do_,ho_,offsets.size()*sizeof(std::uint64_t),cudaMemcpyHostToDevice,stream_)); CUDA_CHECK(cudaEventRecord(e1_,stream_));
        segmented_reduce_kernel<<<(unsigned)seg,threads_,threads_*2*sizeof(double),stream_>>>(dv_,do_,ds_,dm_,dc_,seg); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(e2_,stream_));
        CUDA_CHECK(cudaMemcpyAsync(hs_,ds_,seg*sizeof(double),cudaMemcpyDeviceToHost,stream_)); CUDA_CHECK(cudaMemcpyAsync(hm_,dm_,seg*sizeof(double),cudaMemcpyDeviceToHost,stream_)); CUDA_CHECK(cudaMemcpyAsync(hc_,dc_,seg*sizeof(std::uint64_t),cudaMemcpyDeviceToHost,stream_)); CUDA_CHECK(cudaEventRecord(e3_,stream_)); CUDA_CHECK(cudaEventSynchronize(e3_));
        float a=0,b=0,c=0; CUDA_CHECK(cudaEventElapsedTime(&a,e0_,e1_)); CUDA_CHECK(cudaEventElapsedTime(&b,e1_,e2_)); CUDA_CHECK(cudaEventElapsedTime(&c,e2_,e3_));
        Result r; r.sums.assign(hs_,hs_+seg); r.maxs.assign(hm_,hm_+seg); r.counts.assign(hc_,hc_+seg); r.timing.host_pack_ms=std::chrono::duration<double,std::milli>(p1-p0).count(); r.timing.h2d_ms=a; r.timing.kernel_ms=b; r.timing.d2h_ms=c; return r;
    }
private:
    void release(){ if(dc_)cudaFree(dc_); if(dm_)cudaFree(dm_); if(ds_)cudaFree(ds_); if(do_)cudaFree(do_); if(dv_)cudaFree(dv_); if(hc_)cudaFreeHost(hc_); if(hm_)cudaFreeHost(hm_); if(hs_)cudaFreeHost(hs_); if(ho_)cudaFreeHost(ho_); if(hv_)cudaFreeHost(hv_); dc_=nullptr;dm_=ds_=dv_=nullptr;do_=nullptr;hc_=nullptr;hm_=hs_=hv_=nullptr;ho_=nullptr;vc_=oc_=sc_=0; }
    void ensure(std::size_t v,std::size_t o,std::size_t s){ if(v<=vc_&&o<=oc_&&s<=sc_)return; release(); vc_=std::max<std::size_t>(v,1);oc_=std::max<std::size_t>(o,2);sc_=std::max<std::size_t>(s,1);
        CUDA_CHECK(cudaMallocHost((void**)&hv_,vc_*sizeof(double))); CUDA_CHECK(cudaMallocHost((void**)&ho_,oc_*sizeof(std::uint64_t))); CUDA_CHECK(cudaMallocHost((void**)&hs_,sc_*sizeof(double))); CUDA_CHECK(cudaMallocHost((void**)&hm_,sc_*sizeof(double))); CUDA_CHECK(cudaMallocHost((void**)&hc_,sc_*sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMalloc((void**)&dv_,vc_*sizeof(double))); CUDA_CHECK(cudaMalloc((void**)&do_,oc_*sizeof(std::uint64_t))); CUDA_CHECK(cudaMalloc((void**)&ds_,sc_*sizeof(double))); CUDA_CHECK(cudaMalloc((void**)&dm_,sc_*sizeof(double))); CUDA_CHECK(cudaMalloc((void**)&dc_,sc_*sizeof(std::uint64_t))); }
    int threads_; std::size_t vc_=0,oc_=0,sc_=0; cudaStream_t stream_{}; cudaEvent_t e0_{},e1_{},e2_{},e3_{};
    double *hv_=nullptr,*hs_=nullptr,*hm_=nullptr,*dv_=nullptr,*ds_=nullptr,*dm_=nullptr; std::uint64_t *ho_=nullptr,*hc_=nullptr,*do_=nullptr,*dc_=nullptr;
};
