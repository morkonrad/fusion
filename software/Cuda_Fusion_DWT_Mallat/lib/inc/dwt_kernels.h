#ifndef DWT_KERNELS_H
#define DWT_KERNELS_H

//CUDA Runtime
#include <cuda_runtime.h>

//CUDA Helper functions
#include "helper_cuda.h"
#include "helper_functions.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4200)
#pragma warning(disable : 4996)
#endif

#define BLOCK_DIM 32

//Params DWT_X_GPU
#define X_RESULT_STEPS 4
#define X_BLOCKDIM_X 16
#define X_BLOCKDIM_Y 4

//Params DWT_Y_GPU
#define Y_RESULT_STEPS 4
#define Y_BLOCKDIM_X 8
#define Y_BLOCKDIM_Y 16

#define I_X_RESULT_STEPS 3
#define I_X_BLOCKDIM_X 32
#define I_X_BLOCKDIM_Y 4

#define I_Y_RESULT_STEPS 3
#define I_Y_BLOCKDIM_X 4
#define I_Y_BLOCKDIM_Y 32

//#define _CUDA_OFF

#ifdef _CUDA_OFF
static void set_filters(float* lpd, float* hpd, float* lpr, float* hpr){};
static void DWT_X_GPU(float *d_ip, const int &rows, const int &cols, const int &cA_cols, const int &f_len, float *d_cL, float *d_cH, const int &halo, cudaStream_t stream){};
static void DWT_Y_GPU(float *d_ip, const int &rows, const int &cols, const int &cA_rows, const int &f_len, float *d_cL, float *d_cH, const int &halo, cudaStream_t stream){};
static void IDWT_X_GPU_1(float *d_dst, float*d_src_A, float *d_src_D, const int &rows, const int &cols, const int &next_cols, const int &filt_len, cudaStream_t stream){};
static void IDWT_Y_GPU_1(float *d_dst, float*d_src_A, float *d_src_D, const int &rows, const int &cols, const int &next_rows, const int &filt_len, cudaStream_t stream){};
static void DWT_compare(float *d_ip_v, float *d_ip_ir, int app_len, int out_len, cudaStream_t stream1, cudaStream_t stream2){};
#else
void set_filters(float* lpd, float* hpd, float* lpr, float* hpr);
void DWT_X_GPU(float *d_ip, const int &rows, const int &cols, const int &cA_cols, const int &f_len, float *d_cL, float *d_cH, const int &halo, cudaStream_t stream);
void DWT_Y_GPU(float *d_ip, const int &rows, const int &cols, const int &cA_rows, const int &f_len, float *d_cL, float *d_cH, const int &halo, cudaStream_t stream);
void IDWT_X_GPU_1(float *d_dst, float*d_src_A, float *d_src_D, const int &rows, const int &cols, const int &next_cols, const int &filt_len, cudaStream_t stream);
void IDWT_Y_GPU_1(float *d_dst, float*d_src_A, float *d_src_D, const int &rows, const int &cols, const int &next_rows, const int &filt_len, cudaStream_t stream);
void DWT_compare(float *d_ip_v, float *d_ip_ir, int app_len, int out_len, cudaStream_t stream1, cudaStream_t stream2);
#endif

#endif // DWT_KERNELS_H
