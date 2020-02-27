#ifndef FUSION_H
#define FUSION_H

#include <vector>
#include <cstdint>
#include "cuda_runtime_api.h"
#include "common.h"
#include "dwt_lift.h"


namespace FUSE_CUDA
{
    class Fusion_DWT
    {
    private:

        DATATYPE* _dev_image_visual;
        DATATYPE* _dev_image_ir;

        DATATYPE* _dev_analysis_out_visual;
        DATATYPE* _dev_analysis_out_ir;

        DATATYPE* _dev_analysis_fused;
        DATATYPE* _dev_synthesis_out_pyramid_fused;

        DATATYPE* _dev_synthesis_out_visual;
        DATATYPE* _dev_synthesis_out_ir;

        void* _pined_in_vis;
        void* _pined_in_ir;

        bool _integrated_gpu;

        std::vector<DATATYPE> _host_analysis_out_vis;
        std::vector<DATATYPE> _host_analysis_out_ir;

        std::uint32_t _img_cnt_rows;
        std::uint32_t _img_cnt_cols;
        std::uint64_t _img_size_bytes_in;
        std::uint32_t _cnt_levels;

        std::uint64_t _pyramid_size_bytes_reconstruction;
        std::uint64_t _img_size_bytes_analysis_lvl0;

        std::uint64_t _synthesis_extra_elements_size;
        std::uint64_t _img_size_elements_analysis_lvl0;


        cudaStream_t _stream_io_from_gpu;
        cudaStream_t _stream_io_to_gpu;
        cudaStream_t _stream_compute_vis;
        cudaStream_t _stream_compute_ir;

        DWT_LIFT::dwt_lift _dwt;

    public:

        Fusion_DWT( const uint8_t *host_imgVis, const uint8_t *host_imgIR,
                const std::uint32_t& rows,
                const std::uint32_t& cols,
                const std::uint32_t& levels,
                const std::uint64_t& img_size_bytes);

        ~Fusion_DWT();

        int calc_analysis_async( const std::uint8_t* host_imgVis,
                           const std::uint8_t* host_imgIR,
                           const std::uint32_t& img_cols,
                           const std::uint32_t& img_rows,
                           const std::uint64_t& img_size_bytes);

        int calc_analysis_sync( const std::uint8_t& lvl_id,
                           const std::uint8_t* host_imgVis,
                           const std::uint8_t* host_imgIR,
                           const std::uint32_t& img_cols,
                           const std::uint32_t& img_rows,
                           const std::uint64_t& img_size_bytes);

        int get_analysis_results( std::vector<DATATYPE>& analysis_results_vis,
                                  std::vector<DATATYPE>& analysis_results_ir)const;

        const std::vector<DATATYPE>*
        get_analysis_vis()const;

        const std::vector<DATATYPE>*
        get_analysis_ir()const;

        int get_analysis_fused(std::vector<DATATYPE> &analysis_results_fused)const;

        int get_fused_input(std::vector<DATATYPE> &results_fused)const;

    };


}



#endif



