#ifndef F_WAVELET_H
#define F_WAVELET_H


#include "wavefilt.h"
#include "dwt_kernels.h"
#include <cinttypes>
#include <assert.h>

//NPP libraries
#include <npp.h>


typedef struct wave_set* wave_object;

struct wave_set {
    char wname[50];
    int filtlength;// When all filters are of the same length. [Matlab uses zero-padding to make all filters of the same length]

    float *lpd;
    float *hpd;
    float *lpr;
    float *hpr;

    float params[0];

};

typedef struct wt_set* wt_object;

struct wt_set {

    int siglength;	// Length of the original signal.
    int outlength;	// Length of the output DWT vector
    int lenlength;  // Length of the Output Dimension Vector "length"
    int n_levels;			// Number of decomposition Levels                  $(NVCUDASAMPLES_ROOT)/common/inc
    int MaxIter;	// Maximum Iterations J <= MaxIter

    int rows, cols;
    int length[102][2];   //Length of 2d images at different stages
};


class F_Wavelet
{

public:

    wave_object wave_obj;
    wt_object wt_fus;

    F_Wavelet(const int &rows, const int &cols, const char *wave, const int &n_levels)
    {
        Initialize(rows,cols,wave,n_levels);
    }

    ~F_Wavelet()
    {
        //wave_free();
        //dev_mem_free();
        wt_free();
    }


    void wave_reinit(const char *wname);

    void set_n_lev(int n_levels);

    float* Fuse_RGBA(const std::uint8_t *Img_v, const std::uint8_t*Img_ir,const uint& width, const uint& height);
    float* Fuse_Grayscale(const std::uint8_t *Img_v, const std::uint8_t *Img_ir,const uint& width, const uint& height);


    void dwt2(std::vector<float>& ir_analysis,std::vector<float>& vis_analysis);

private:

    cudaStream_t stream1, stream2;

    float *fused;
    float *d_ip_v, *d_cL_v, *d_cH_v;
    float *d_ip_ir, *d_cL_ir, *d_cH_ir;

    float *d_lpd, *d_hpd, *d_lpr, *d_hpr;

    Npp8u *Image_V_C, *Image_IR_C, *Image_V_Gr, *Image_IR_Gr;
    Npp32f *Image_V_F,*Image_IR_F;

    std::uint32_t _alloc_size_vis;
    std::uint32_t _alloc_size_ir;

    NppiSize ROI;
    const Npp32f coef[3] = { 0.2125f , 0.7154f, 0.0721f };

    int step_C, step_G, step_F;

    int check_iter(const int & rows, const int & cols, const int & J);

    void wave_init(const char *wname);

    void wt_init_dummy(const int & rows, const int & cols, const int & J);

    void cu_filt_alloc();

    void dwt();



    void idwt();

    void Initialize(const int &rows, const int &cols, const char *wave, const int &n_levels);

    void wave_free();

    void dev_mem_free();

    void wt_free();

};

#endif // F_WAVELET_H
