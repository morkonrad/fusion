#pragma once

//CUDA Runtime
#include <cuda_runtime.h>

//CUDA Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <assert.h>

//GPU sep_rows 
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 4

//GPU sep_columns
#define COLUMNS_BLOCKDIM_X 4
#define COLUMNS_BLOCKDIM_Y 16
#define COLUMNS_RESULT_STEPS 4

//Conv Block dimensions (16x16)
#define   CONV_BLOCKDIM 16
#define   CONV_RESULT_STEPS 3 // 3 steps in both x and y dimensions

// Expand and Reduce
#define BLOCKDIM_X 32
#define BLOCKDIM_Y 16

//Transpose & Recursive gaussian block dim
#define BLOCK_DIM 32

//Recursive Gaussian edges clamping
#define CLAMP_TO_EDGE 1

//#define _CUDA_OFF

#ifdef _CUDA_OFF

static void set_filter(float *d_Kernel){};
static void CU_subtract(float * d_dst, float * d_Src_1, float * d_Src_2, const int &len, cudaStream_t stream){};
static void CU_add(float * d_dst, float * d_lap, const int &len){};
static void compare_Lp(float *d_ip_v, float *d_ip_ir, int app_len, int out_len, cudaStream_t stream1, cudaStream_t stream2){};
static void convolutionRowsGPU_v1(float *d_Dst, float *d_Src, int imageW, int imageH, int filter_Rad, int Halo_steps){};
static void convolutionColumnsGPU_v1(float *d_Dst, float *d_Src, int imageW, int imageH, int filter_Rad, int Halo_steps){};
static void convolutionRowsGPU_down_smp(float *d_Dst,float *d_Src,int imageW,int n_imageW,int imageH,int filter_Rad,int Halo_steps,cudaStream_t stream){};
static void convolutionColumnsGPU_down_smp(float *d_Dst, float *d_Src, int imageW, int imageH, int n_imageH, int filter_Rad, int Halo_steps, cudaStream_t stream){};
static void convolutionRowsGPU_up_smp(float *d_Dst, float *d_Src, int imageW, int n_imageW, int imageH, int filter_Rad, int Halo_steps, cudaStream_t stream){};
static void convolutionColumnsGPU_up_smp(float *d_Dst,float *d_Src,int imageW,int imageH,int n_imageH,int filter_Rad,int Halo_steps,cudaStream_t stream){};
static void convolutionRecursive_down_smp_1(float * d_dst, float * d_src, float* d_buffer, int imageW, int imageH, int cA_h, int cA_w){};
static void convolutionRecursive_up_smp_1(float * d_dst, float * d_src, float* d_buffer, int imageW, int imageH, int next_W, int next_H){};
static void transpose(float *d_src, float *d_dest, int width, int height){};

#else

void set_filter(float *d_Kernel);
void CU_subtract(float * d_dst, float * d_Src_1, float * d_Src_2, const int &len,cudaStream_t stream);
void CU_add(float * d_dst, float * d_lap, const int &len);

void compare_Lp(float *d_ip_v, float *d_ip_ir, int app_len, int out_len,cudaStream_t stream1,cudaStream_t stream2);

//void convolutionRowsGPU_v1(float *d_Dst, float *d_Src, int imageW, int imageH, int filter_Rad, int Halo_steps);
//void convolutionColumnsGPU_v1(float *d_Dst, float *d_Src, int imageW, int imageH, int filter_Rad, int Halo_steps);


void convolutionRowsGPU_down_smp(float *d_Dst,float *d_Src,int imageW,int n_imageW,int imageH,int filter_Rad,int Halo_steps,cudaStream_t stream);
void convolutionColumnsGPU_down_smp(float *d_Dst, float *d_Src, int imageW, int imageH, int n_imageH, int filter_Rad, int Halo_steps,cudaStream_t stream);

void convolutionRowsGPU_up_smp(float *d_Dst, float *d_Src, int imageW, int n_imageW, int imageH, int filter_Rad, int Halo_steps,cudaStream_t stream);
void convolutionColumnsGPU_up_smp(float *d_Dst,float *d_Src,int imageW,int imageH,int n_imageH,int filter_Rad,int Halo_steps,cudaStream_t stream);


void convolutionRecursive_down_smp_1(float * d_dst, float * d_src, float* d_buffer, int imageW, int imageH, int cA_h, int cA_w,cudaStream_t stream);
void convolutionRecursive_up_smp_1(float * d_dst, float * d_src, float* d_buffer, int imageW, int imageH, int next_W, int next_H,cudaStream_t stream);

void transpose(float *d_src, float *d_dest, int width, int height);

#endif
