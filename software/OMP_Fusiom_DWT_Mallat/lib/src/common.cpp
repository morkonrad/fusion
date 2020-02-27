/// ****************************************************************************
/// ***************** Common utilities and  CUDA Kernels  **********************
/// ****************************************************************************

//~ #include "utils.h"
#include "common.h"

template <typename T> T W_SIGN(T val) {
    return (T{0} < val) ?  T{1} :  T{-1};
}

//#define W_SIGN(a) ((a > 0) ? (1.0f) : (-1.0f))
#define SQRT_2 1.4142135623730951
//#include <cublas.h>
#include <cmath>
#include <algorithm>

DTYPE c_kern_L[MAX_FILTER_WIDTH] __attribute__((aligned(16)));
DTYPE c_kern_H[MAX_FILTER_WIDTH] __attribute__((aligned(16)));
DTYPE c_kern_IL[MAX_FILTER_WIDTH] __attribute__((aligned(16)));
DTYPE c_kern_IH[MAX_FILTER_WIDTH] __attribute__((aligned(16)));

DTYPE c_kern_LL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH] __attribute__((aligned(16)));
DTYPE c_kern_LH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH] __attribute__((aligned(16)));
DTYPE c_kern_HL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH] __attribute__((aligned(16)));
DTYPE c_kern_HH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH] __attribute__((aligned(16)));



/// soft thresholding of the detail coefficients (2D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
void w_kern_soft_thresh(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_h[row*Nc + col];
        c_h[row*Nc + col] = std::copysign(std::max(std::abs(val)-beta, DTYPE{0}), val);

        val = c_v[row*Nc + col];
        c_v[row*Nc + col] = std::copysign(std::max(std::abs(val)-beta, DTYPE{0}), val);

        val = c_d[row*Nc + col];
        c_d[row*Nc + col] = std::copysign(std::max(std::abs(val)-beta, DTYPE{0}), val);
 		  
	  }
  }
  
/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }*/
}

/// soft thresholding of the detail coefficients (1D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
// CHECKME: consider merging this kernel with the previous kernel
void w_kern_soft_thresh_1d(DTYPE* c_d, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_d[row*Nc + col];
        c_d[row*Nc + col] = std::copysign(std::max(std::abs(val)-beta, DTYPE{0}), val);
 		  
	  }
  }

  
/*  
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
*/
}

/// soft thresholding of the approximation coefficients (2D and 1D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
void w_kern_soft_thresh_appcoeffs(DTYPE* c_a, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_a[row*Nc + col];
        c_a[row*Nc + col] = std::copysign(std::max(std::abs(val)-beta, DTYPE{0}), val);
 		  
	  }
  }

   
/*   
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = copysignf(max(fabsf(val)-beta, 0.0f), val);
    }
*/
}


/// Hard thresholding of the detail coefficients (2D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
void w_kern_hard_thresh(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc) {


  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_h[row*Nc + col];
        c_h[row*Nc + col] = std::max(W_SIGN(std::abs(val)-beta), DTYPE{0})*val;
        val = c_v[row*Nc + col];
        c_v[row*Nc + col] = std::max(W_SIGN(std::abs(val)-beta), DTYPE{0})*val;
        val = c_d[row*Nc + col];
        c_d[row*Nc + col] = std::max(W_SIGN(std::abs(val)-beta), DTYPE{0})*val; 		  
	  }
  }

 
 /*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
*/
}


/// Hard thresholding of the detail coefficients (1D)
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
// CHECKME: consider merging this kernel with the previous kernel
void w_kern_hard_thresh_1d(DTYPE* c_d, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_d[row*Nc + col];
        c_d[row*Nc + col] = std::max(W_SIGN(std::abs(val)-beta), DTYPE{0})*val;
	  }
  }

/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
*/
 
}


/// Hard thresholding of the approximation coefficients (2D and 1D)
void w_kern_hard_thresh_appcoeffs(DTYPE* c_a, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_a[row*Nc + col];
        c_a[row*Nc + col] = std::max(W_SIGN(std::abs(val)-beta), DTYPE{0})*val;
	  }
  }

    
/*    
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = max(W_SIGN(fabsf(val)-beta), 0.0f)*val;
    }
*/
}

/// Projection of the coefficients onto the L-infinity ball of radius "beta" (2D).
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
void w_kern_proj_linf(DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_h[row*Nc + col];
        c_h[row*Nc + col] = std::copysign(std::min(std::abs(val),beta), val);

        val = c_v[row*Nc + col];
        c_v[row*Nc + col] = std::copysign(std::min(std::abs(val),beta), val);

        val = c_d[row*Nc + col];
        c_d[row*Nc + col] = std::copysign(std::min(std::abs(val),beta), val);
 		  
	  }
  }
 
/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_h[gidy*Nc + gidx];
        c_h[gidy*Nc + gidx] = copysignf(min(fabsf(val), beta), val);

        val = c_v[gidy*Nc + gidx];
        c_v[gidy*Nc + gidx] = copysignf(min(fabsf(val), beta), val);

        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(min(fabsf(val), beta), val);
    }
*/
}

void w_kern_proj_linf_appcoeffs(DTYPE* c_a, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_a[row*Nc + col];
        c_a[row*Nc + col] = std::copysign(std::min(std::abs(val),beta), val);
	  }
  }
 
/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_a[gidy*Nc + gidx];
        c_a[gidy*Nc + gidx] = copysignf(min(fabsf(val), beta), val);
    }
*/
}

/// Projection of the coefficients onto the L-infinity ball of radius "beta" (1D).
/// Must be lanched with block size (Nc, Nr) : the size of the current coefficient vector
void w_kern_proj_linf_1d(DTYPE* c_d, DTYPE beta, int Nr, int Nc) {

  DTYPE val = 0.0f;
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        val = c_d[row*Nc + col];
        c_d[row*Nc + col] = std::copysign(std::min(std::abs(val),beta), val);
	  }
  }

/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    DTYPE val = 0.0f;
    if (gidx < Nc && gidy < Nr) {
        val = c_d[gidy*Nc + gidx];
        c_d[gidy*Nc + gidx] = copysignf(min(fabsf(val), beta), val);
    }
*/
}





/// Circular shift of the image (2D and 1D)
void w_kern_circshift(DTYPE* d_image, DTYPE* d_out, int Nr, int Nc, int sr, int sc) {

  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
        int r = row - sr, c = col - sc;
        if (r < 0) r += Nr;
        if (c < 0) c += Nc;
        d_out[row*Nc + col] = d_image[r*Nc + c];
	  }
  }

/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidx < Nc && gidy < Nr) {
        int r = gidy - sr, c = gidx - sc;
        if (r < 0) r += Nr;
        if (c < 0) c += Nc;
        d_out[gidy*Nc + gidx] = d_image[r*Nc + c];
    }
*/
}



/// ****************************************************************************
/// ******************** Common CUDA Kernels calls *****************************
/// ****************************************************************************

void w_call_soft_thresh(/* DTYPE** */ wavelet_coeff_t& d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs, int normalize) {

    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) w_div2(&Nr2);
        w_div2(&Nc2);
    }
    if (do_thresh_appcoeffs) {
        DTYPE beta2 = beta;
        if (normalize > 0) { // beta2 = beta/sqrt(2)^nlevels
            int nlevels2 = nlevels/2;
            beta2 /= (1 << nlevels2);
            if (nlevels2 *2 != nlevels) beta2 /= SQRT_2;
        }
        w_kern_soft_thresh_appcoeffs(d_coeffs[0].data(), beta2, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) w_div2(&Nr);
            w_div2(&Nc);
        }
        if (normalize > 0) beta /= SQRT_2;
        if (ndims > 1) w_kern_soft_thresh(d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), beta, Nr, Nc);
        else w_kern_soft_thresh_1d(d_coeffs[i+1].data(), beta, Nr, Nc);
    }


  
/*  
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) w_div2(&Nr2);
        w_div2(&Nc2);
    }
    if (do_thresh_appcoeffs) {
        DTYPE beta2 = beta;
        if (normalize > 0) { // beta2 = beta/sqrt(2)^nlevels
            int nlevels2 = nlevels/2;
            beta2 /= (1 << nlevels2);
            if (nlevels2 *2 != nlevels) beta2 /= SQRT_2;
        }
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_soft_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta2, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) w_div2(&Nr);
            w_div2(&Nc);
        }
        if (normalize > 0) beta /= SQRT_2;
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        if (ndims > 1) w_kern_soft_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
        else w_kern_soft_thresh_1d<<<n_blocks, n_threads_per_block>>>(d_coeffs[i+1], beta, Nr, Nc);
    }
*/ 
    
}


void w_call_hard_thresh(/* DTYPE** */ wavelet_coeff_t&  d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs, int normalize) {

    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) w_div2(&Nr2);
        w_div2(&Nc2);
    }
    DTYPE beta2 = beta;
    if (do_thresh_appcoeffs) {
        if (normalize > 0) { // beta2 = beta/sqrt(2)^nlevels
            int nlevels2 = nlevels/2;
            beta2 /= (1 << nlevels2);
            if (nlevels2 *2 != nlevels) beta2 /= SQRT_2;
        }
        w_kern_hard_thresh_appcoeffs(d_coeffs[0].data(), beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) w_div2(&Nr);
            w_div2(&Nc);
        }
        if (normalize > 0) beta /= SQRT_2;
        if (ndims > 1) w_kern_hard_thresh(d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), beta, Nr, Nc);
        else w_kern_hard_thresh_1d(d_coeffs[i+1].data(), beta, Nr, Nc);
    }

/*
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) w_div2(&Nr2);
        w_div2(&Nc2);
    }
    DTYPE beta2 = beta;
    if (do_thresh_appcoeffs) {
        if (normalize > 0) { // beta2 = beta/sqrt(2)^nlevels
            int nlevels2 = nlevels/2;
            beta2 /= (1 << nlevels2);
            if (nlevels2 *2 != nlevels) beta2 /= SQRT_2;
        }
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_hard_thresh_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) w_div2(&Nr);
            w_div2(&Nc);
        }
        if (normalize > 0) beta /= SQRT_2;
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        if (ndims > 1) w_kern_hard_thresh<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
        else w_kern_hard_thresh_1d<<<n_blocks, n_threads_per_block>>>(d_coeffs[i+1], beta, Nr, Nc);
    }

*/
}


void w_call_proj_linf(/* DTYPE** */ wavelet_coeff_t&  d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs) {

    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) w_div2(&Nr2);
        w_div2(&Nc2);
    }
    if (do_thresh_appcoeffs) {
        w_kern_proj_linf_appcoeffs(d_coeffs[0].data(), beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) w_div2(&Nr);
            w_div2(&Nc);
        }
        if (ndims > 1) w_kern_proj_linf(d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), beta, Nr, Nc);
        else w_kern_proj_linf_1d(d_coeffs[i+1].data(), beta, Nr, Nc);
    }
  
/*  
    int tpb = 16; // Threads per block
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels, ndims = winfos.ndims;
    int Nr2 = Nr, Nc2 = Nc;
    if (!do_swt) {
        if (ndims > 1) w_div2(&Nr2);
        w_div2(&Nc2);
    }
    if (do_thresh_appcoeffs) {
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_proj_linf_appcoeffs<<<n_blocks, n_threads_per_block>>>(d_coeffs[0], beta, Nr2, Nc2);
    }
    for (int i = 0; i < nlevels; i++) {
        if (!do_swt) {
            if (ndims > 1) w_div2(&Nr);
            w_div2(&Nc);
        }
        n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
        if (ndims > 1) w_kern_proj_linf<<<n_blocks, n_threads_per_block>>>(d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], beta, Nr, Nc);
        else w_kern_proj_linf_1d<<<n_blocks, n_threads_per_block>>>(d_coeffs[i+1], beta, Nr, Nc);
    }
    
*/   
}




void w_shrink(/* DTYPE** */ wavelet_coeff_t&  d_coeffs, DTYPE beta, w_info winfos, int do_thresh_appcoeffs) {
    int  nlevels = winfos.nlevels, ndims = winfos.ndims;

//    int Nr2 = Nr, Nc2 = Nc;

//    if (!do_swt) {
//        if (ndims > 1) w_div2(&Nr2);
//        w_div2(&Nc2);
//    }
    DTYPE scale_factor = 1.0f/(1.0f + beta);
    
    if (do_thresh_appcoeffs) {
		
        //cublas_scal(Nr2*Nc2, 1.0f/(1.0f + beta), d_coeffs[0], 1);
        std::transform(d_coeffs[0].begin(),d_coeffs[0].end(),d_coeffs[0].begin(),[=](DTYPE val) {return val*scale_factor;});
    }
    for (int i = 0; i < nlevels; i++) {
//        if (!do_swt) {
//            if (ndims > 1) w_div2(&Nr);
//            w_div2(&Nc);
//        }
        if (ndims == 2) {

           std::transform(d_coeffs[3*i+1].begin(),d_coeffs[3*i+1].end(),d_coeffs[3*i+1].begin(),[=](DTYPE val) {return val*scale_factor;});
           std::transform(d_coeffs[3*i+2].begin(),d_coeffs[3*i+2].end(),d_coeffs[3*i+2].begin(),[=](DTYPE val) {return val*scale_factor;});
           std::transform(d_coeffs[3*i+3].begin(),d_coeffs[3*i+3].end(),d_coeffs[3*i+3].begin(),[=](DTYPE val) {return val*scale_factor;});
			
//            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+1], 1);
//            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+2], 1);
//            cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[3*i+3], 1);
        }
        else { // 1D
           std::transform(d_coeffs[i+1].begin(),d_coeffs[i+1].end(),d_coeffs[i+1].begin(),[=](DTYPE val) {return val*scale_factor;});
           // cublas_scal(Nr*Nc, 1.0f/(1.0f + beta), d_coeffs[i+1], 1);
        }
    }
}





// if inplace = 1, the result is in "d_image" ; otherwise result is in "d_image2".
void w_call_circshift(DTYPE* d_image, DTYPE* d_image2, w_info winfos, int sr, int sc, int inplace) {

    int Nr = winfos.Nr, Nc = winfos.Nc, ndims = winfos.ndims;
    // Modulus in C can be negative
    if (sr < 0) sr += Nr; // or do while loops to ensure positive numbers
    if (sc < 0) sc += Nc;
    //int tpb = 16; // Threads per block
    sr = sr % Nr;
    sc = sc % Nc;
    if (ndims == 1) sr = 0;
    if (inplace) {
        w_kern_circshift(d_image2, d_image, Nr, Nc, sr, sc);
    }
    else {
        w_kern_circshift(d_image, d_image2, Nr, Nc, sr, sc);
    }
   
   
/*   
    int Nr = winfos.Nr, Nc = winfos.Nc, ndims = winfos.ndims;
    // Modulus in C can be negative
    if (sr < 0) sr += Nr; // or do while loops to ensure positive numbers
    if (sc < 0) sc += Nc;
    int tpb = 16; // Threads per block
    sr = sr % Nr;
    sc = sc % Nc;
    if (ndims == 1) sr = 0;
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    if (inplace) {
        cudaMemcpy(d_image2, d_image, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        w_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image2, d_image, Nr, Nc, sr, sc);
    }
    else {
        w_kern_circshift<<<n_blocks, n_threads_per_block>>>(d_image, d_image2, Nr, Nc, sr, sc);
    }
*/
}


/// Creates an allocated/padded device array : [ An, H1, V1, D1, ..., Hn, Vn, Dn]
/*
DTYPE** w_create_coeffs_buffer(w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    int Nr0 = Nr, Nc0 = Nc;
    if (!do_swt) {
        w_div2(&Nr0);
        w_div2(&Nc0);
    }
    DTYPE** res = (DTYPE**) calloc(3*nlevels+1, sizeof(DTYPE*));
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            w_div2(&Nr);
            w_div2(&Nc);
        }
        cudaMalloc(&(res[i]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i], 0, Nr*Nc*sizeof(DTYPE));
        cudaMalloc(&(res[i+1]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i+1], 0, Nr*Nc*sizeof(DTYPE));
        cudaMalloc(&(res[i+2]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i+2], 0, Nr*Nc*sizeof(DTYPE));
    }
    // App coeff (last scale). They are also useful as a temp. buffer for the reconstruction, hence a bigger size
    cudaMalloc(&(res[0]), Nr0*Nc0*sizeof(DTYPE));
    cudaMemset(res[0], 0, Nr0*Nc0*sizeof(DTYPE));

    return res;
}


/// Creates an allocated/padded device array : [ An, D1, ..., Dn]
DTYPE** w_create_coeffs_buffer_1d(w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    int Nc0 = Nc;
    if (!do_swt) w_div2(&Nc0);
    DTYPE** res = (DTYPE**) calloc(nlevels+1, sizeof(DTYPE*));
    // Det coeffs
    for (int i = 1; i < nlevels+1; i++) {
        if (!do_swt) w_div2(&Nc);
        cudaMalloc(&(res[i]), Nr*Nc*sizeof(DTYPE));
        cudaMemset(res[i], 0, Nr*Nc*sizeof(DTYPE));
    }
    // App coeff (last scale). They are also useful as a temp. buffer for the reconstruction, hence a bigger size
    cudaMalloc(&(res[0]), Nr*Nc0*sizeof(DTYPE));
    cudaMemset(res[0], 0, Nr*Nc0*sizeof(DTYPE));
    return res;
}



/// Deep free of wavelet coefficients
void w_free_coeffs_buffer(DTYPE** coeffs, int nlevels) {
    for (int i = 0; i < 3*nlevels+1; i++) cudaFree(coeffs[i]);
    free(coeffs);
}

void w_free_coeffs_buffer_1d(DTYPE** coeffs, int nlevels) {
    for (int i = 0; i < nlevels+1; i++) cudaFree(coeffs[i]);
    free(coeffs);
}
*/

/// Deep copy of wavelet coefficients. All structures must be allocated.
/*
void w_copy_coeffs_buffer(DTYPE** dst, DTYPE** src, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels, do_swt = winfos.do_swt;
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            w_div2(&Nr);
            w_div2(&Nc);
        }
        cudaMemcpy(dst[i], src[i], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+1], src[i+1], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst[i+2], src[i+2], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // App coeff (last scale)
    cudaMemcpy(dst[0], src[0], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
}


void w_copy_coeffs_buffer_1d(DTYPE** dst, DTYPE** src, w_info winfos) {
    int Nr = winfos.Nr, Nc = winfos.Nc, nlevels = winfos.nlevels, do_swt = winfos.do_swt;
    // Det Coeffs
    for (int i = 1; i < nlevels+1; i++) {
        if (!do_swt) w_div2(&Nc);
        cudaMemcpy(dst[i], src[i], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // App coeff (last scale)
    cudaMemcpy(dst[0], src[0], Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
}

*/

///
/// ----------------------------------------------------------------------------
///



/*
void w_add_coeffs(DTYPE** dst, DTYPE** src, w_info winfos, DTYPE alpha) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    // Coeffs (H, V, D)
    for (int i = 1; i < 3*nlevels+1; i += 3) {
        if (!do_swt) {
            w_div2(&Nr);
            w_div2(&Nc);
        }
        cublas_axpy(Nr*Nc, alpha, src[i], 1, dst[i], 1);
        cublas_axpy(Nr*Nc, alpha, src[i+1], 1, dst[i+1], 1);
        cublas_axpy(Nr*Nc, alpha, src[i+2], 1, dst[i+2], 1);
    }
    // App coeff (last scale)
    cublas_axpy(Nr*Nc, alpha, src[0], 1, dst[0], 1);
}


/// dst = dst + alpha*src
void w_add_coeffs_1d(DTYPE** dst, DTYPE** src, w_info winfos, DTYPE alpha) {
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    // Det Coeffs
    for (int i = 1; i < nlevels+1; i++) {
        if (!do_swt) Nc /= 2;
        cublas_axpy(Nr*Nc, alpha, src[i], 1, dst[i], 1);
    }
    // App coeff (last scale)
    cublas_axpy(Nr*Nc, alpha, src[0], 1, dst[0], 1);
}
*/


