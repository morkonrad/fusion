#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

//#include <cuda.h>
//#include <cublas.h>
//#include <cuComplex.h>

//~ #include "utils.h"
#include "common.h"
#include "wt.h"
#include "separable.h"
#include "nonseparable.h"
#include "haar.h"

/*
#  define CUDACHECK \
  { cudaThreadSynchronize(); \
    cudaError_t last = cudaGetLastError();\
    if(last!=cudaSuccess) {\
      printf("ERRORX: %s  %s  %i \n", cudaGetErrorString( last),    __FILE__, __LINE__    );    \
      exit(1);\
    }\
  }
*/

// FIXME: temp. workaround
#define MAX_FILTER_WIDTH 40



/// ****************************************************************************
/// ******************** Wavelets class ****************************************
/// ****************************************************************************


/// Constructor : copy assignment
// do not use !
/*
Wavelets& Wavelets::operator=(const Wavelets &rhs) {
  if (this != &rhs) { // protect against invalid self-assignment
    // allocate new memory and copy the elements
    size_t sz = rhs.Nr * rhs.Nc * sizeof(DTYPE);
    DTYPE* new_image, *new_tmp;
    DTYPE** new_coeffs;
    cudaMalloc(&new_image, sz);
    cudaMemcpy(new_image, rhs.d_image, sz, cudaMemcpyDeviceToDevice);

    new_coeffs =  w_create_coeffs_buffer(rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);
    if (ndim == 2) w_copy_coeffs_buffer(new_coeffs, rhs.coeffs, rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);
    else  w_copy_coeffs_buffer_1d(new_coeffs, rhs.coeffs, rhs.Nr, rhs.Nc, rhs.nlevels, rhs.do_swt);

    cudaMalloc(&new_tmp, sz);
    cudaMemcpy(new_tmp, rhs.d_tmp, 2*sz, cudaMemcpyDeviceToDevice); // Two temp. images

    // deallocate old memory
    cudaFree(d_image);
    w_free_coeffs_buffer(coeffs, nlevels);
    cudaFree(d_tmp);
    // assign the new memory to the object
    d_image = new_image;
    coeffs = new_coeffs;
    d_tmp = new_tmp;
    Nr = rhs.Nr;
    Nc = rhs.Nc;
    strncpy(wname, rhs.wname, 128);
    nlevels = rhs.nlevels;
    do_cycle_spinning = rhs.do_cycle_spinning;
    current_shift_r = rhs.current_shift_r;
    current_shift_c = rhs.current_shift_c;
    do_swt = rhs.do_swt;
    do_separable = rhs.do_separable;
  }
  return *this;
}
*/


/// Constructor : default
Wavelets::Wavelets(void) : d_image(NULL),d_tmp(NULL),current_shift_r(0), current_shift_c(0), do_separable(1),do_cycle_spinning(0)
{
}


/// Constructor :  Wavelets from image
Wavelets::Wavelets(
    DTYPE* img,
    int Nr,
    int Nc,
    const char* wname,
    int levels,
    int memisonhost,
    int do_separable,
    int do_cycle_spinning,
    int do_swt,
    int ndim) :

    d_image(NULL),
    d_tmp(NULL),
    current_shift_r(0),
    current_shift_c(0),
    do_separable(do_separable),
    do_cycle_spinning(do_cycle_spinning),
    state(W_INIT)
{
    winfos.Nr = Nr;
    winfos.Nc = Nc;
    winfos.nlevels = levels;
    winfos.do_swt = do_swt;
    winfos.ndims = ndim;
    
    if(do_separable)
    {
      w_separable = std::make_unique<separable_wavelet_transform>(winfos);
      w_nonseparable = nullptr;
    }
    else
    {
      w_nonseparable = std::make_unique<nonseparable_wavelet_transform>(winfos);
      w_separable = nullptr;
    }  

    if (levels < 1) {
        puts("Warning: cannot initialize wavelet coefficients with nlevels < 1. Forcing nlevels = 1");
        winfos.nlevels = 1;
    }

    // Image
   /* DTYPE* d_arr_in;
    cudaMalloc(&d_arr_in, Nr*Nc*sizeof(DTYPE));
    if (!img) cudaMemset(d_arr_in, 0, Nr*Nc*sizeof(DTYPE));
    else {
        cudaMemcpyKind transfer;
        if (memisonhost) transfer = cudaMemcpyHostToDevice;
        else transfer = cudaMemcpyDeviceToDevice;
        cudaMemcpy(d_arr_in, img, Nr*Nc*sizeof(DTYPE), transfer);
    }*/
   
    image.assign(img,img+Nr*Nc);
    d_image = image.data();

   // DTYPE* d_tmp_new;
   // cudaMalloc(&d_tmp_new, 2*Nr*Nc*sizeof(DTYPE)); // Two temp. images
    tmp.resize(2*Nr*Nc);
    d_tmp = tmp.data();

    // Dimensions
    if (Nr == 1) { // 1D data
        ndim = 1;
        winfos.ndims = 1;
    }

    if (ndim == 1 && do_separable == 0) {
        puts("Warning: 1D DWT was requestred, which is incompatible with non-separable transform.");
        puts("Ignoring the do_separable option.");
        do_separable = 1;
    }
    
    /*
    // Filters
    strncpy(this->wname, wname, 128);
    int hlen = 0;
    if (do_separable) hlen = w_compute_filters_separable(wname, do_swt);
    else hlen = w_compute_filters(wname, 1, do_swt);
    if (hlen == 0) {
        printf("ERROR: unknown wavelet name %s\n", wname);
        //~ exit(1);
        state = W_CREATION_ERROR;
    }
    winfos.hlen = hlen;

    // Compute max achievable level according to image dimensions and filter size
    int N;
    if (ndim == 2) N = min(Nr, Nc);
    else N = Nc;
    int wmaxlev = w_ilog2(N/hlen);
    // TODO: remove this limitation
    if (levels > wmaxlev) {
        printf("Warning: required level (%d) is greater than the maximum possible level for %s (%d) on a %dx%d image.\n", winfos.nlevels, wname, wmaxlev, winfos.Nc, winfos.Nr);
        printf("Forcing nlevels = %d\n", wmaxlev);
        winfos.nlevels = wmaxlev;
    }
    */
    // Allocate coeffs
    //DTYPE** coeffs_new;
    if (ndim == 1) allocate_coeffs_1D();
    else if (ndim == 2) allocate_coeffs_2D();
    else {
        printf("ERROR: ndim=%d is not implemented\n", ndim);
        //~ exit(1);
        //~ throw std::runtime_error("Error on ndim");
        state = W_CREATION_ERROR;
    }
    //coeffs = coeffs_new;
    if (do_cycle_spinning && do_swt) puts("Warning: makes little sense to use Cycle spinning with stationary Wavelet transform");
    // TODO
    if (do_cycle_spinning && ndim == 1) {
        puts("ERROR: cycle spinning is not implemented for 1D. Use SWT instead.");
        //~ exit(1);
        state = W_CREATION_ERROR;
    }

}






/// Constructor: copy
Wavelets::Wavelets(const Wavelets &W) :
    current_shift_r(W.current_shift_r),
    current_shift_c(W.current_shift_c),
    do_separable(W.do_separable),
    do_cycle_spinning(W.do_cycle_spinning),
    state(W.state)
{
    winfos.Nr = W.winfos.Nr;
    winfos.Nc = W.winfos.Nc;
    winfos.nlevels = W.winfos.nlevels;
    winfos.ndims = W.winfos.ndims;
    winfos.hlen = W.winfos.hlen;
    winfos.do_swt = W.winfos.do_swt;

    strncpy(wname, W.wname, 128);

    image.assign(W.image.begin(),W.image.end());

    tmp.resize(W.tmp.size());
    d_tmp = tmp.data();

    coeffs.resize(W.coeffs.size());
    for(size_t i=0;i<coeffs.size();++i)
    {
	   coeffs[i] = W.coeffs[i];
	}
    
    if (winfos.ndims == 1) {
    }
    else if (winfos.ndims == 2) {
    }
    else {
        puts("ERROR: 3D wavelets not implemented yet");
        state = W_CREATION_ERROR;
    }
}


/// Destructor
Wavelets::~Wavelets() 
{
}

void Wavelets::set_filters(const char* wname)
{
	strncpy(this->wname, wname, 128);
    int hlen = 0;
    if (do_separable) hlen =  w_separable->set_filters(wname,winfos.do_swt); //w_compute_filters_separable(wname, winfos.do_swt);
    else hlen = w_nonseparable->set_filters(wname,winfos.do_swt);//w_compute_filters(wname, 1, winfos.do_swt);
    if (hlen == 0) {
        std::cout <<" ERROR: unknown wavelet name " << wname << std::endl;
        //printf("ERROR: unknown wavelet name %s\n", wname);
        //~ exit(1);
        state = W_CREATION_ERROR;
    }
    winfos.hlen = hlen;

    // Compute max achievable level according to image dimensions and filter size
    int N;
    if (winfos.ndims == 2) N = std::min(winfos.Nr, winfos.Nc);
    else N = winfos.Nc;
    int wmaxlev = w_ilog2(N/hlen);
    // TODO: remove this limitation
    if (winfos.nlevels  > wmaxlev) {
        std::cout << " Warning: required level is greater than the maximum possible level " << std::endl;
        std::cout << "Forcing nlevels = " << wmaxlev << std::endl;

        //printf("Warning: required level (%d) is greater than the maximum possible level for %s (%d) on a %dx%d image.\n", winfos.nlevels, wname, wmaxlev, winfos.Nc, winfos.Nr);
        //printf("Forcing nlevels = %d\n", wmaxlev);
        winfos.nlevels = wmaxlev;
    }
}

void Wavelets::allocate_coeffs_1D()
{
   int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
   int Nc0 = Nc;
   if (!do_swt) w_div2(&Nc0);
    
   coeffs.resize(nlevels+1); 
   for (int i = 1; i < nlevels+1; i++) 
   {
       if (!do_swt) w_div2(&Nc);
       coeffs[i].resize(Nr*Nc);
       std::fill(coeffs[i].begin(),coeffs[i].end(),(DTYPE)0);
   }  

   coeffs[0].resize(Nr*Nc0);
   std::fill(coeffs[0].begin(),coeffs[0].end(),(DTYPE)0);

}

void Wavelets::allocate_coeffs_2D()
{
    int Nr = winfos.Nr, Nc = winfos.Nc, do_swt = winfos.do_swt, nlevels = winfos.nlevels;
    int Nr0 = Nr, Nc0 = Nc;
    if (!do_swt) {
        w_div2(&Nr0);
        w_div2(&Nc0);
    }

   coeffs.resize(3*nlevels+1); 
   for (int i = 1; i < 3*nlevels+1; i += 3) 
   {
       if (!do_swt) {
            w_div2(&Nr);
            w_div2(&Nc);
        }
        for(int j =0;j<3;++j)
        {
		    coeffs[i+j].resize(Nr*Nc); 
		    std::fill(coeffs[i+j].begin(),coeffs[i+j].end(),(DTYPE)0);
		}        
   }  
   coeffs[0].resize(Nr0*Nc0);
   std::fill(coeffs[0].begin(),coeffs[0].end(),(DTYPE)0);


}

 



/// Method : forward
void Wavelets::forward_2D(void) {
    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    // TODO: handle W_FORWARD_ERROR with return codes of transforms
    if (do_cycle_spinning) {
        current_shift_r = rand() % winfos.Nr;
        current_shift_c = rand() % winfos.Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_forward2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->forward_2D(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->forward_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
   
    
    // else: not implemented yet
    state = W_FORWARD;

}


void Wavelets::forward_2D_even_symmetric(void) {
    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    // TODO: handle W_FORWARD_ERROR with return codes of transforms
    if (do_cycle_spinning) {
        current_shift_r = rand() % winfos.Nr;
        current_shift_c = rand() % winfos.Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_forward2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->forward_2D_even_symmetric(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->forward_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
   
    
    // else: not implemented yet
    state = W_FORWARD;

}

void Wavelets::forward_2D_even_symmetric_cdf_97_wavelet()
{
	    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    // TODO: handle W_FORWARD_ERROR with return codes of transforms
    if (do_cycle_spinning) {
        current_shift_r = rand() % winfos.Nr;
        current_shift_c = rand() % winfos.Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_forward2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->forward_2D_even_symmetric_cdf_97_wavelet(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->forward_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
   
    
    // else: not implemented yet
    state = W_FORWARD;
}    

void Wavelets::forward_2D_even_symmetric_cdf_53_wavelet()
{
	    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    // TODO: handle W_FORWARD_ERROR with return codes of transforms
    if (do_cycle_spinning) {
        current_shift_r = rand() % winfos.Nr;
        current_shift_c = rand() % winfos.Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
    
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_forward2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->forward_2D_even_symmetric_cdf_53_wavelet(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->forward_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->forward_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
   
    
    // else: not implemented yet
    state = W_FORWARD;
}    


void Wavelets::forward_1D(void) {
    if (state == W_CREATION_ERROR) {
        puts("Warning: forward transform not computed, as there was an error when creating the wavelets");
        return;
    }
    // TODO: handle W_FORWARD_ERROR with return codes of transforms
    if (do_cycle_spinning) {
        current_shift_r = rand() % winfos.Nr;
        current_shift_c = rand() % winfos.Nc;
        circshift(current_shift_r, current_shift_c, 1);
    }
 
    if ((winfos.hlen == 2) && (!winfos.do_swt))
    {
		 haar_forward1d(d_image, coeffs, d_tmp, winfos);
         return;
    }		 
    
    if (winfos.do_swt)
    {
		 w_separable->forward_swt_1D(d_image, coeffs);
	}
    else 
    {
		w_separable->forward_1D(d_image, coeffs);
    }
 
 
    state = W_FORWARD;

}


/// Method : inverse
void Wavelets::inverse_2D(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if (state == W_FORWARD_ERROR || state == W_THRESHOLD_ERROR) {
        puts("Warning: inverse transform not computed, as there was an error in a previous stage");
        return;
    }
  
  
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_inverse2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->inverse_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->inverse_2D(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->inverse_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->inverse_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
    // else: not implemented yet
    if (do_cycle_spinning) circshift(-current_shift_r, -current_shift_c, 1);
    state = W_INVERSE;
}

void Wavelets::inverse_2D_even_symmetric_cdf_97_wavelet(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if (state == W_FORWARD_ERROR || state == W_THRESHOLD_ERROR) {
        puts("Warning: inverse transform not computed, as there was an error in a previous stage");
        return;
    }
  
  
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_inverse2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->inverse_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->inverse_2D_even_symmetric_cdf_97_wavelet(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->inverse_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->inverse_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
    // else: not implemented yet
    if (do_cycle_spinning) circshift(-current_shift_r, -current_shift_c, 1);
    state = W_INVERSE;
}

void Wavelets::inverse_2D_even_symmetric_cdf_53_wavelet(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if (state == W_FORWARD_ERROR || state == W_THRESHOLD_ERROR) {
        puts("Warning: inverse transform not computed, as there was an error in a previous stage");
        return;
    }
  
  
    
   if ((winfos.hlen == 2) && (!winfos.do_swt))
   {
	    haar_inverse2d(d_image, coeffs, d_tmp, winfos);
        return;  
   }    
   
   if(do_separable)
   {
	   if(winfos.do_swt)
	   {
		   w_separable->inverse_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_separable->inverse_2D_even_symmetric_cdf_53_wavelet(d_image, coeffs);		   
	   }
	   return; 
   }
   else
   {
	   if(winfos.do_swt)
	   {
		   w_nonseparable->inverse_swt_2D(d_image, coeffs);
	   }
	   else
	   {
		   w_nonseparable->inverse_2D(d_image, coeffs);		   
	   }
	   return; 
	   
   }
   
    // else: not implemented yet
    if (do_cycle_spinning) circshift(-current_shift_r, -current_shift_c, 1);
    state = W_INVERSE;
}



void Wavelets::inverse_1D(void) {
    if (state == W_INVERSE) { // TODO: what to do in this case ? Force re-compute, or abort ?
        puts("Warning: W.inverse() has already been run. Inverse is available in W.get_image()");
        return;
    }
    if (state == W_FORWARD_ERROR || state == W_THRESHOLD_ERROR) {
        puts("Warning: inverse transform not computed, as there was an error in a previous stage");
        return;
    }
    // TODO: handle W_INVERSE_ERROR with return codes of inverse transforms
 
    if ((winfos.hlen == 2) && (!winfos.do_swt))
    {
	   haar_inverse1d(d_image, coeffs, d_tmp, winfos);
       return;		
    }
 
    if (winfos.do_swt)
    {
		 w_separable->inverse_swt_1D(d_image, coeffs);
	}
    else 
    {
		w_separable->inverse_1D(d_image, coeffs);
    }
 
    
    
    // else: not implemented yet
    if (do_cycle_spinning) circshift(-current_shift_r, -current_shift_c, 1);
    state = W_INVERSE;
}


/// Method : soft thresholding (L1 proximal)
void Wavelets::soft_threshold(DTYPE beta, int do_thresh_appcoeffs, int normalize) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_soft_thresh(coeffs, beta, winfos, do_thresh_appcoeffs, normalize);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}

/// Method : hard thresholding
void Wavelets::hard_threshold(DTYPE beta, int do_thresh_appcoeffs, int normalize) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_hard_thresh(coeffs, beta, winfos, do_thresh_appcoeffs, normalize);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}

/// Method : shrink (L2 proximal)
void Wavelets::shrink(DTYPE beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_shrink(coeffs, beta, winfos, do_thresh_appcoeffs);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}
/// Method : projection onto the L-infinity ball (infinity norm proximal, i.e dual L1 norm proximal)
void Wavelets::proj_linf(DTYPE beta, int do_thresh_appcoeffs) {
    if (state == W_INVERSE) {
        puts("Warning: Wavelets(): cannot threshold coefficients, as they were modified by W.inverse()");
        return;
    }
    w_call_proj_linf(coeffs, beta, winfos, do_thresh_appcoeffs);
    // TODO: handle W_THRESHOLD_ERROR from a return code
}





/// Method : circular shift
// If inplace = 1, the result is in d_image ; otherwise result is in d_tmp.
void Wavelets::circshift(int sr, int sc, int inplace) {
    w_call_circshift(d_image, d_tmp, winfos, sr, sc, inplace);
}
/// Method : squared L2 norm
DTYPE Wavelets::norm2sq(void) {


    DTYPE res = 0.0f;
    
    for (int i = 0; i < winfos.nlevels; i++) {
        if (winfos.ndims == 2) { // 2D
            
            res += std::accumulate(coeffs[3*i+1].begin(),coeffs[3*i+1].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+b*b;});
            res += std::accumulate(coeffs[3*i+2].begin(),coeffs[3*i+2].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+b*b;});
            res += std::accumulate(coeffs[3*i+3].begin(),coeffs[3*i+3].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+b*b;});
        }
        else { // 1D
            res += std::accumulate(coeffs[i+1].begin(),coeffs[i+1].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+std::abs(b);});
        }
    }

    res += std::accumulate(coeffs[0].begin(),coeffs[0].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+b*b;});

    return res;

/*
    DTYPE res = 0.0f;
    int Nr2 = winfos.Nr;
    int Nc2 = winfos.Nc;
    DTYPE tmp = 0;
    for (int i = 0; i < winfos.nlevels; i++) {
        if (!winfos.do_swt) {
            if (winfos.ndims > 1) w_div2(&Nr2);
            w_div2(&Nc2);
        }
        if (winfos.ndims == 2) { // 2D
            tmp = cublas_nrm2(Nr2*Nc2, coeffs[3*i+1], 1);
            res += tmp*tmp;
            tmp =cublas_nrm2(Nr2*Nc2, coeffs[3*i+2], 1);
            res += tmp*tmp;
            tmp = cublas_nrm2(Nr2*Nc2, coeffs[3*i+3], 1);
            res += tmp*tmp;
        }
        else { // 1D
            res += cublas_asum(Nr2*Nc2, coeffs[i+1], 1);
        }
    }
    tmp = cublas_nrm2(Nr2*Nc2, coeffs[0], 1);
    res += tmp*tmp;
    return res;
*/

}

/// Method : L1 norm
DTYPE Wavelets::norm1(void) {


    DTYPE res = 0.0f;
    
    for (int i = 0; i < winfos.nlevels; i++) {
        if (winfos.ndims == 2) { // 2D
            
            res += std::accumulate(coeffs[3*i+1].begin(),coeffs[3*i+1].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+std::abs(b);});
            res += std::accumulate(coeffs[3*i+2].begin(),coeffs[3*i+2].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+std::abs(b);});
            res += std::accumulate(coeffs[3*i+3].begin(),coeffs[3*i+3].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+std::abs(b);});
        }
        else { // 1D
            res += std::accumulate(coeffs[i+1].begin(),coeffs[i+1].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+std::abs(b);});
        }
    }

    res += std::accumulate(coeffs[0].begin(),coeffs[0].end(),(DTYPE)0,[](DTYPE a,DTYPE b){return a+std::abs(b);});

    return res;


/*
    DTYPE res = 0.0f;
    int Nr2 = winfos.Nr;
    int Nc2 = winfos.Nc;
    for (int i = 0; i < winfos.nlevels; i++) {
        if (!winfos.do_swt) {
            if (winfos.ndims > 1) w_div2(&Nr2);
            w_div2(&Nc2);
        }
        if (winfos.ndims == 2) { // 2D
            res += cublas_asum(Nr2*Nc2, coeffs[3*i+1], 1);
            res += cublas_asum(Nr2*Nc2, coeffs[3*i+2], 1);
            res += cublas_asum(Nr2*Nc2, coeffs[3*i+3], 1);
        }
        else { // 1D
            res += cublas_asum(Nr2*Nc2, coeffs[i+1], 1);
        }
    }
    res += cublas_asum(Nr2*Nc2, coeffs[0], 1);
    return res;
*/
}

/// Method : get the image from device
int Wavelets::get_image(DTYPE* res) { // TODO: more defensive
	
	std::copy(image.begin(),image.end(),res);
    //cudaMemcpy(res, d_image, winfos.Nr*winfos.Nc*sizeof(DTYPE), cudaMemcpyDeviceToHost);
    return winfos.Nr*winfos.Nc;
}

/// Method : set the class image
void Wavelets::set_image(DTYPE* img, int mem_is_on_device) { // There are no memory check !
    
    image.assign(img,img+winfos.Nr*winfos.Nc);
    /*
    cudaMemcpyKind copykind;
    if (mem_is_on_device) copykind = cudaMemcpyDeviceToDevice;
    else copykind = cudaMemcpyHostToDevice;
    cudaMemcpy(d_image, img, winfos.Nr*winfos.Nc*sizeof(DTYPE), copykind);
    */
    
    state = W_INIT;
}


/// Method : set a coefficient
void Wavelets::set_coeff(DTYPE* coeff, int num, int mem_is_on_device) { // There are no memory check !
 
  std::copy(coeff,coeff+coeffs[num].size(),coeffs[num].begin());
   
 /*
    cudaMemcpyKind copykind;
    if (mem_is_on_device) copykind = cudaMemcpyDeviceToDevice;
    else copykind = cudaMemcpyHostToDevice;
    int Nr2 = winfos.Nr, Nc2 = winfos.Nc;
    if (winfos.ndims == 2) {
        // In 2D, num stands for the following:
        // A  H1 V1 D1  H2 V2 D2
        // 0  1  2  3   4  5  6
        // for num>0,  1+(num-1)/3 tells the scale number
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = ((num-1)/3) +1;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nr2);
            w_div2(&Nc2);
        }
    }
    else if (winfos.ndims == 1) {
        // In 1D, num stands for the following:
        // A  D1 D2 D3
        // 0  1  2  3
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = num;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nc2);
        }
    }
    cudaMemcpy(coeffs[num], coeff, Nr2*Nc2*sizeof(DTYPE), copykind);
    //~ state = W_FORWARD; // ?
*/
}





/// Method : get a coefficient vector from device
int Wavelets::get_coeff(DTYPE* coeff, int num) {
    if (state == W_INVERSE) {
        puts("Warning: get_coeff(): inverse() has been performed, the coefficients has been modified and do not make sense anymore.");
        return 0;
    }
    std::copy(coeffs[num].begin(),coeffs[num].end(),coeff);
    return coeffs[num].size(); 
    /*
    int Nr2 = winfos.Nr, Nc2 = winfos.Nc;
    if (winfos.ndims == 2) {
        // In 2D, num stands for the following:
        // A  H1 V1 D1  H2 V2 D2
        // 0  1  2  3   4  5  6
        // for num>0,  1+(num-1)/3 tells the scale number
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = ((num-1)/3) +1;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nr2);
            w_div2(&Nc2);
        }
    }
    else if (winfos.ndims == 1) {
        // In 1D, num stands for the following:
        // A  D1 D2 D3
        // 0  1  2  3
        int scale;
        if (num == 0) scale = winfos.nlevels;
        else scale = num;
        if (!winfos.do_swt) for (int i = 0; i < scale; i++) {
            w_div2(&Nc2);
        }
    }
    //~ printf("Retrieving %d (%d x %d)\n", num, Nr2, Nc2);
    cudaMemcpy(coeff, coeffs[num], Nr2*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToHost); //TODO: handle DeviceToDevice ?
    * */
   // return Nr2*Nc2;
}



/// Method : give some informations on the wavelet
void Wavelets::print_informations() {

    const char* state[2] = {"no", "yes"};
    puts("------------- Wavelet transform infos ------------");
    printf("Data dimensions : ");
    if (winfos.ndims == 2) printf("(%d, %d)\n", winfos.Nr, winfos.Nc);
    else { // 1D
        if (winfos.Nr == 1) printf("%d\n", winfos.Nc);
        else printf("(%d, %d) [batched 1D transform]\n", winfos.Nr, winfos.Nc);
    }
    printf("Wavelet name : %s\n", wname);
    printf("Number of levels : %d\n", winfos.nlevels);
    printf("Stationary WT : %s\n", state[winfos.do_swt]);
    printf("Cycle spinning : %s\n", state[do_cycle_spinning]);
    printf("Separable transform : %s\n", state[do_separable]);

    size_t mem_used = 0;
    if (!winfos.do_swt) {
        // DWT : size(output) = size(input), since sizes are halved at each level.
        // d_image (1), coeffs (1), d_tmp (2)
        mem_used = 5*winfos.Nr*winfos.Nc*sizeof(DTYPE);
    }
    else {
        // SWT : size(output) = size(input)*4*levels
        // d_image (1), coeffs (3*levels+1), d_tmp (2)
        if (winfos.ndims == 2) mem_used = (3*winfos.nlevels+4)*winfos.Nr*winfos.Nc*sizeof(DTYPE);
        else mem_used = (winfos.nlevels+4)*winfos.Nr*winfos.Nc*sizeof(DTYPE);
    }
    printf("Estimated memory footprint : %.2f MB\n", mem_used/1e6);


 /*   int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    char* device_name = strdup(properties.name);
    printf("Running on device : %s\n", device_name);
    free(device_name);*/
    puts("--------------------------------------------------");
}


/// Provide a custom filter bank to the current Wavelet instance.
/// If do_separable = 1, the filters are expected to be L, H.
/// Otherwise, the filters are expected to be A, H, V, D (square size)
// We cannot directly use the __constant__ symbols (unless with separate compilation),
// hence a further indirection in (non)separable.cu where these symbols are defined
int Wavelets::set_filters_forward(char* filtername, uint len, DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4) {
    int res = 0;
    if (len > MAX_FILTER_WIDTH) {
        printf("ERROR: Wavelets.set_filters_forward(): filter length (%d) exceeds the maximum size (%d)\n", len, MAX_FILTER_WIDTH);
        return -1;
    }
    if (do_separable) {
		//std::cout << "set forward filter coeff" << std::endl;
        //res = w_set_filters_forward(filter1, filter2, len);
        res = w_separable->set_filters_forward(filter1, filter2, len);
    }
    else {
        if (filter3 == NULL || filter4 == NULL) {
            puts("ERROR: Wavelets.set_filters_forward(): expected argument 4 and 5 for non-separable filtering");
            return -2;
        }
        res = w_nonseparable->set_filters_forward(filter1, filter2, filter3, filter4, len);
    }
    winfos.hlen = len;
    strncpy(wname, filtername, 128);

    return res;
}

/// Here the filters are assumed to be of the same size of those provided to set_filters_forward()
// We cannot directly use the __constant__ symbols (unless with separate compilation),
// hence a further indirection in (non)separable.cu where these symbols are defined
int Wavelets::set_filters_inverse(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4) {
    uint len = winfos.hlen;
    int res = 0;
    if (do_separable) {
        // ignoring args 4 and 5
       // res = w_set_filters_inverse(filter1, filter2, len);
        res = w_separable->set_filters_inverse(filter1, filter2, len);

    }
    else {
        if (filter3 == NULL || filter4 == NULL) {
            puts("ERROR: Wavelets.set_filters_inverse(): expected argument 4 and 5 for non-separable filtering");
            return -2;
        }
        // The same symbols are used for the inverse filters
        res = w_nonseparable->set_filters_inverse(filter1, filter2, filter3, filter4, len);
    }

    return res;
}




/// ----------------------------------------------------------------------------
/// --------- Operators... for now I am not considering overloading  -----------
/// ----------------------------------------------------------------------------


/**
 * \brief In-place addition of wavelets coefficients
 *
 *  For a given instance "Wav" of the class Wavelets, it performs
 *   Wav += W. Only the wavelets coefficients are added, the image attribute
 *  is not replaced
 *
 *
 * \param W : Wavelets class instance
 * \return 0 if no error
 *
 */
/* 
int Wavelets::add_wavelet(Wavelets W, DTYPE alpha) {

    // Various checks
    if ((winfos.nlevels != W.winfos.nlevels) || (strcasecmp(wname, W.wname))) {
        puts("ERROR: add_wavelet(): right operand is not the same transform (wname, level)");
        return -1;
    }
    if (state == W_INVERSE || W.state == W_INVERSE) {
        puts("WARNING: add_wavelet(): this operation makes no sense when wavelet has just been inverted");
        return 1;
    }
    if (winfos.Nr != W.winfos.Nr || winfos.Nc != W.winfos.Nc || winfos.ndims != W.winfos.ndims) {
        puts("ERROR: add_wavelet(): operands do not have the same geometry");
        return -2;
    }
    if ((winfos.do_swt) ^ (W.winfos.do_swt)) {
        puts("ERROR: add_wavelet(): operands should both use SWT or DWT");
        return -3;
    }
    if (
        (do_cycle_spinning * W.do_cycle_spinning)
        && (
            (current_shift_r != W.current_shift_r) || (current_shift_c != W.current_shift_c)
           )
       )
    {
        puts("ERROR: add_wavelet(): operands do not have the same current shift");
        return -4;
    }

    if (winfos.ndims == 1) w_adcoeffs_1d(coeffs, W.coeffs, winfos, alpha);
    else w_adcoeffs(coeffs, W.coeffs, winfos, alpha);
    return 0;
}

*/











