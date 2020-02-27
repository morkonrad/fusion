#include "fusion.h"
#include "fusion_rule.h"
#include <iostream>

using namespace FUSE_CUDA;

inline void check_driver_error(cudaError err,int line)
{
    if(err!=cudaError_t::cudaSuccess)
    {
        std::cerr<<"driver_error at line: "<<line<<" in File: "<<__FILE__<<std::endl;
        std::cerr<<"Error: "<<cudaGetErrorString(err)<<" EXIT !"<<std::endl;
        std::exit(-1);
    }

}

Fusion_DWT::Fusion_DWT(const uint8_t *host_imgVis,
                       const uint8_t *host_imgIR,
                       const uint32_t &rows,
                       const uint32_t &cols,
                       const uint32_t &levels,
                       const std::uint64_t& img_size_bytes)
{
    if(rows*cols==0||levels==0)std::runtime_error("");


    _img_cnt_rows = rows;
    _img_cnt_cols = cols;
    _cnt_levels = levels;
    _img_size_bytes_in = img_size_bytes;

    //TODO: Modify the assumption that the target platform includes only a single GPU
    _integrated_gpu = false;

    int nDevices;
    auto err = cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if(prop.integrated!=0)_integrated_gpu=true;
/*
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);        
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
*/
    }
/*
    if(_integrated_gpu)
        std::cout<<"Detcted iGPU"<<std::endl;
    else
        std::cout<<"Detcted dGPU"<<std::endl;
*/
    cudaError ok ;

    if(_integrated_gpu)
    {
        ok = cudaMallocHost(&_pined_in_vis,_img_size_bytes_in);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif
        ok = cudaMallocHost(&_pined_in_ir,_img_size_bytes_in);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif

    }
    else
    {
        _pined_in_vis = (void*)host_imgVis;
        _pined_in_ir = (void*)host_imgIR;

        ok = cudaHostRegister(_pined_in_vis,_img_size_bytes_in,cudaHostRegisterPortable);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif

        ok = cudaHostRegister(_pined_in_ir,_img_size_bytes_in,cudaHostRegisterPortable);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif
    }

    ok = cudaStreamCreate(&_stream_compute_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaStreamCreate(&_stream_compute_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaStreamCreate(&_stream_io_from_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaStreamCreate(&_stream_io_to_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    _host_analysis_out_ir.resize(_img_cnt_cols*_img_cnt_rows,0);
    _host_analysis_out_vis.resize(_img_cnt_cols*_img_cnt_rows,0);

    const size_t input_img_size_bytes  = _img_cnt_cols*_img_cnt_rows*sizeof(DATATYPE);

    auto cnt_elem_pyramid = 0;

    for(int i=1; i<_cnt_levels; ++i)
    {
        const auto ccols = (_img_cnt_cols/(2<<(i-1)));
        const auto crows = (_img_cnt_rows/(2<<(i-1)));
        cnt_elem_pyramid +=  ccols*crows;
    }

    _img_size_elements_analysis_lvl0 = _img_cnt_cols*_img_cnt_rows;
    _synthesis_extra_elements_size = cnt_elem_pyramid;

    _pyramid_size_bytes_reconstruction = input_img_size_bytes + (cnt_elem_pyramid* sizeof(DATATYPE));
    _img_size_bytes_analysis_lvl0 = input_img_size_bytes;

    ok = cudaMalloc ((void**) &_dev_image_visual, _img_size_bytes_analysis_lvl0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif
    ok = cudaMalloc ((void**) &_dev_analysis_out_visual, _img_size_bytes_analysis_lvl0 );
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaMalloc ((void**) &_dev_synthesis_out_visual, _pyramid_size_bytes_reconstruction);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaMalloc ((void**) &_dev_image_ir, _img_size_bytes_analysis_lvl0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaMalloc ((void**) &_dev_analysis_out_ir, _img_size_bytes_analysis_lvl0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaMalloc ((void**) &_dev_synthesis_out_ir, _pyramid_size_bytes_reconstruction);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif


    ok = cudaMalloc ((void**) &_dev_analysis_fused, _img_size_bytes_analysis_lvl0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaMalloc ((void**) &_dev_synthesis_out_pyramid_fused, _pyramid_size_bytes_reconstruction);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

}

Fusion_DWT::~Fusion_DWT()
{

    if(_pined_in_ir!=nullptr)
    {
        if(_integrated_gpu)
        {
            auto ok = cudaFreeHost(_pined_in_ir);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif
        }
        else
        {

            auto ok = cudaHostUnregister((void*)_pined_in_ir);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif

        }
    }


    if(_pined_in_vis!=nullptr)
    {
        if(_integrated_gpu)
        {
            auto ok = cudaFreeHost(_pined_in_vis);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif
        }
        else
        {
            auto ok = cudaHostUnregister((void*)_pined_in_vis);
#ifndef NDEBUG
        check_driver_error(ok,__LINE__);
#endif
        }
    }

    auto ok = cudaStreamDestroy(_stream_compute_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaStreamDestroy(_stream_compute_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaStreamDestroy(_stream_io_from_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaStreamDestroy(_stream_io_to_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_image_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_image_visual);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_analysis_out_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_analysis_out_visual);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_synthesis_out_visual);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_synthesis_out_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_analysis_fused);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif

    ok = cudaFree(_dev_synthesis_out_pyramid_fused);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#endif
}

int Fusion_DWT::calc_analysis_async( const std::uint8_t *host_imgVis,
                                     const std::uint8_t *host_imgIR,
                                     const uint32_t &img_cols,
                                     const uint32_t &img_rows,
                                     const std::uint64_t& size_bytes)
{
    //check if provided host_ptr are compatible with locked-pined ptr and if img_sizes match
    if(host_imgIR==nullptr)return -1;
    if(host_imgVis==nullptr)return -1;

    if(_pined_in_ir==nullptr)return -1;
    if(_pined_in_vis==nullptr)return -1;    


    if( (img_cols!=_img_cnt_cols)||
            (img_rows!=_img_cnt_rows)||
            (size_bytes!=_img_size_bytes_in))
    {
        //TODO:: implement reallocation of memory on the GPU
        return -1;
    }

    if(_integrated_gpu)
    {
        memcpy(_pined_in_vis, host_imgVis, size_bytes);
        memcpy(_pined_in_ir, host_imgIR, size_bytes);
    }
    else
    {
        if(host_imgIR!=_pined_in_ir)return -1;
        if(host_imgVis!=_pined_in_vis)return -1;
    }

    cudaEvent_t event_transfer_to_gpu_vis;
    auto ok = cudaEventCreateWithFlags(&event_transfer_to_gpu_vis,cudaEventDisableTiming);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    cudaEvent_t event_transfer_to_gpu_ir;
    ok = cudaEventCreateWithFlags(&event_transfer_to_gpu_ir,cudaEventDisableTiming);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    cudaEvent_t event_compute_analysis_vis;
    ok = cudaEventCreateWithFlags(&event_compute_analysis_vis,cudaEventDisableTiming);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    cudaEvent_t event_compute_analysis_ir;
    ok = cudaEventCreateWithFlags(&event_compute_analysis_ir,cudaEventDisableTiming);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif


    //---------------------------------------------------
    // non blocking transfer + event record  +transform imgVIS
    //---------------------------------------------------
    ok = cudaMemcpyAsync(_dev_image_visual,_pined_in_vis,size_bytes,cudaMemcpyHostToDevice,_stream_io_to_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaEventRecord(event_transfer_to_gpu_vis,_stream_io_to_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaStreamWaitEvent(_stream_compute_vis,event_transfer_to_gpu_vis,0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    //Call the Transform  on the imgVIS
    ok = _dwt.calc_forward(_cnt_levels,
                           _img_cnt_cols,
                           _img_cnt_rows,
                           _dev_image_visual,
                           _dev_analysis_out_visual,
                           _stream_compute_vis);


    ok = cudaEventRecord(event_compute_analysis_vis,_stream_compute_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    //---------------------------------------------------
    // non blocking transfer + event record  + transform imgIR
    //---------------------------------------------------

    ok = cudaMemcpyAsync(_dev_image_ir,_pined_in_ir,size_bytes,cudaMemcpyHostToDevice,_stream_io_to_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaEventRecord(event_transfer_to_gpu_ir,_stream_io_to_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaStreamWaitEvent(_stream_compute_ir,event_transfer_to_gpu_ir,0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif
    //Call the Transform  on the imgIR
    ok = _dwt.calc_forward(_cnt_levels,
                           _img_cnt_cols,
                           _img_cnt_rows,
                           _dev_image_ir,
                           _dev_analysis_out_ir,
                           _stream_compute_ir);

    ok = cudaEventRecord(event_compute_analysis_ir,_stream_compute_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaStreamWaitEvent(_stream_compute_vis,event_compute_analysis_vis,0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaStreamWaitEvent(_stream_compute_vis,event_compute_analysis_ir,0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    /*
    //DBG ONLY
    ok = cudaMemcpyAsync((void*)_host_analysis_out_vis.data(),(void*)_dev_analysis_out_visual,_img_size_bytes_in,cudaMemcpyDeviceToHost,_stream_io_from_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif


    ok = cudaStreamWaitEvent(_stream_io_from_gpu,event_compute_analysis_ir,0);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaMemcpyAsync((void*)_host_analysis_out_ir.data(),
                         (void*)_dev_analysis_out_ir,_img_size_bytes_in,cudaMemcpyDeviceToHost,_stream_io_from_gpu);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif
*/

    //call fuse
    //call inverse
    ok = FUSE_RULE::calc_fuse_avg( _dev_analysis_out_visual,
                                   _dev_analysis_out_ir,
                                   _dev_analysis_fused,
                                   img_rows,img_cols,
                                   _stream_compute_vis);

#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = _dwt.calc_inverse(_cnt_levels,
                           _img_cnt_cols,
                           _img_cnt_rows,
                           _dev_analysis_fused,
                           _dev_synthesis_out_pyramid_fused,
                           _stream_compute_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    //free events
    ok = cudaEventDestroy(event_transfer_to_gpu_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif
    ok = cudaEventDestroy(event_transfer_to_gpu_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaEventDestroy(event_compute_analysis_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif
    ok = cudaEventDestroy(event_compute_analysis_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    return cudaError_t::cudaSuccess;
}

int Fusion_DWT::calc_analysis_sync(const uint8_t &level_id,
                                   const std::uint8_t *host_imgVis,
                                   const std::uint8_t *host_imgIR,
                                   const uint32_t &img_cols,
                                   const uint32_t &img_rows,
                                   const std::uint64_t& size_bytes)
{
    //pre-conditions check
    if(level_id==0||level_id>_cnt_levels)return -1;
    if(host_imgIR==nullptr)return -1;
    if(host_imgVis==nullptr)return -1;


    if( (img_cols!=_img_cnt_cols)||
            (img_rows!=_img_cnt_rows)||
            (size_bytes!=_img_size_bytes_in))
    {
        //TODO:: implement reallocation of memory on the GPU
        return -1;
    }

    if(_integrated_gpu)
    {
        memcpy(_pined_in_vis, host_imgVis, size_bytes);
        memcpy(_pined_in_ir, host_imgIR, size_bytes);
    }


    auto ok = cudaMemcpy(_dev_image_visual,_pined_in_vis,size_bytes,cudaMemcpyHostToDevice);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaMemcpy(_dev_image_ir,_pined_in_ir,size_bytes,cudaMemcpyHostToDevice);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif



    ok = _dwt.calc_forward(level_id,
                           _img_cnt_cols,
                           _img_cnt_rows,
                           _dev_image_visual,
                           _dev_analysis_out_visual,
                           _stream_compute_vis);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = _dwt.calc_forward(level_id,
                           _img_cnt_cols,
                           _img_cnt_rows,
                           _dev_image_ir,
                           _dev_analysis_out_ir,
                           _stream_compute_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = cudaDeviceSynchronize();
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = FUSE_RULE::calc_fuse_avg( _dev_analysis_out_visual,
                                   _dev_analysis_out_ir,
                                   _dev_analysis_fused,
                                   img_rows,img_cols,
                                   _stream_compute_vis);

#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    ok = _dwt.calc_inverse(level_id,
                           _img_cnt_cols,
                           _img_cnt_rows,
                           _dev_analysis_fused,
                           _dev_synthesis_out_pyramid_fused,
                           _stream_compute_ir);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif


    ok = cudaDeviceSynchronize();
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    return cudaError_t::cudaSuccess;
}

int Fusion_DWT::get_fused_input(std::vector<DATATYPE> &results_fused) const
{    
    const auto elems = _img_cnt_cols*_img_cnt_rows;
    results_fused.resize(elems);

    auto ok = cudaMemcpy(results_fused.data(), _dev_synthesis_out_pyramid_fused+_synthesis_extra_elements_size, elems*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif
    return cudaError_t::cudaSuccess;
}

int Fusion_DWT::get_analysis_fused(std::vector<DATATYPE> &analysis_results_fused)const
{
    const auto elems = _img_cnt_cols*_img_cnt_rows;
    analysis_results_fused.resize(elems);

    auto ok = cudaMemcpy(analysis_results_fused.data(), _dev_analysis_fused, elems*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif
    return cudaError_t::cudaSuccess;
}

int Fusion_DWT::get_analysis_results(std::vector<DATATYPE> &analysis_results_vis,
                                     std::vector<DATATYPE> &analysis_results_ir) const
{
    const auto elems = _img_cnt_cols*_img_cnt_rows;
    analysis_results_vis.resize(elems);

    auto ok = cudaMemcpy(analysis_results_vis.data(), _dev_analysis_out_visual, elems*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    analysis_results_ir.resize(elems);

    ok = cudaMemcpy(analysis_results_ir.data(), _dev_analysis_out_ir, elems*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)return ok;
#endif

    return cudaError_t::cudaSuccess;
}

const std::vector<DATATYPE> *
Fusion_DWT::get_analysis_vis() const
{
    auto ok = cudaDeviceSynchronize();
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)
        return nullptr;
#endif

    return &_host_analysis_out_vis;
}

const std::vector<DATATYPE> *
Fusion_DWT::get_analysis_ir() const
{
    auto ok = cudaDeviceSynchronize();
#ifndef NDEBUG
    check_driver_error(ok,__LINE__);
#else
    if(ok!=0)
        return nullptr;
#endif

    return &_host_analysis_out_ir;
}





