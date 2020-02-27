#pragma once

#include <cinttypes>
#include <math.h>

//NPP libraries
#include <npp.h>

#include "Lap_Pyr_Kernels.h"


enum class ePyramidConvolution{ Gauss_separable=1,Gauss_recursive};



class f_Lap_pyr
{
public:
    f_Lap_pyr(int rows, int cols, float sigma, int n_levels, ePyramidConvolution conv)
    {
        LP_init(rows,cols,sigma,n_levels,conv);
    }

    ~f_Lap_pyr()
    {
        LP_free();

        //filt_free();

    }

    void filt_alloc(const float & sigma, const ePyramidConvolution & conv);	 
    float* Fuse_Lap_Pyr_RGBA(const std::uint8_t *Img_v, const std::uint8_t *Img_ir,const uint& width, const uint& height);
    float* Fuse_Lap_Pyr_Grayscale(const std::uint8_t *Img_v, const std::uint8_t *Img_ir,const uint& width, const uint& height);    
	void reset_n_lev(int n_levels);

private:

    cudaStream_t stream1, stream2;

    Npp8u *Image_V_C, *Image_IR_C, *Image_V_Gr, *Image_IR_Gr;
    Npp32f *Image_V_F,*Image_IR_F;

    float *d_lap_ir, *d_input_ir, *d_buffer_ir, *d_reduce_ir, *d_expand_ir;
    float *d_lap_v, *d_input_v, *d_buffer_v, *d_reduce_v, *d_expand_v;

    float *h_filter, *d_filter;

    float *fused;

    NppiSize ROI;
    const Npp32f coef[3] = { 0.2125f , 0.7154f, 0.0721f };

    int step_C, step_G, step_F;

	int img_rows, img_cols; 

    int filter_len;
    ePyramidConvolution conv_methd;

	int signal_len, out_len, scales; 

	int **length;

    void LP_init(int rows, int cols, float sigma, int n_levels, ePyramidConvolution conv);

    void calcfilter(const float &sigma, const ePyramidConvolution &conv);

	void len_calc();

	void set_n_lev(int n_lev); 

    void Laplacian_const_Sep();

    void Image_reconst_Sep();

    void Laplacian_const_Rec();

    void Image_reconst_Rec();

	void dev_free();

    void LP_free();

    void filt_free();

};

