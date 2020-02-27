#ifndef SEPARABLE_H
#define SEPARABLE_H
#include "utils.h"
#include "filters.h"
#include "data_types.h"


class separable_wavelet_transform{
	
	public:
	
	separable_wavelet_transform(w_info winfo);
	~separable_wavelet_transform();
	
    int set_filters(const char* wname, int do_swt);
    int set_filters_forward(DTYPE* filter1, DTYPE* filter2, uint len);
    int set_filters_inverse(DTYPE* filter1, DTYPE* filter2, uint len);
    int forward_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int forward_2D_even_symmetric(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int forward_2D_even_symmetric_cdf_97_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int forward_2D_even_symmetric_cdf_53_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int forward_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs);
    int inverse_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int inverse_2D_even_symmetric_cdf_97_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int inverse_2D_even_symmetric_cdf_53_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int inverse_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs);
    int forward_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int forward_swt_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs);
    int inverse_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int inverse_swt_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs);
	
	private:

    void forward_pass1(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen);
    void forward_pass2(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen);
    void forward_pass1_even_symmetric(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen);
    void forward_pass2_even_symmetric(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen);
    void forward_pass1_even_symmetric_unroll2(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen);
    void forward_pass2_even_symmetric_unroll2(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen);
    void forward_pass1_even_symmetric_unroll4(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen);
    void forward_pass2_even_symmetric_unroll4(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen);
    void forward_pass1_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ img,DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen);
    void forward_pass2_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ tmp_a1, const DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen);
    void forward_pass1_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen);
    void forward_pass2_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ tmp_a1, const DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen);

    void inverse_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int Nr2, int hlen);
    void inverse_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int Nc2, int hlen);
    void inverse_pass1_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ c_a, const DTYPE* __restrict__ c_h, const DTYPE* __restrict__ c_v, const DTYPE* __restrict__ c_d, DTYPE* __restrict__ tmp1, DTYPE* __restrict__ tmp2, int Nr, int Nc, int Nr2, int hlen);
    void inverse_pass2_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ tmp1, const DTYPE* __restrict__ tmp2, DTYPE* img, int Nr, int Nc, int Nc2, int hlen);
    void inverse_pass1_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ c_a, const DTYPE* __restrict__ c_h, const DTYPE* __restrict__ c_v, const DTYPE* __restrict__ c_d, DTYPE* __restrict__ tmp1, DTYPE* __restrict__ tmp2, int Nr, int Nc, int Nr2, int hlen);
    void inverse_pass2_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ tmp1, const DTYPE* __restrict__ tmp2, DTYPE* __restrict__ img, int Nr, int Nc, int Nc2, int hlen);
    void forward_swt_pass1(DTYPE* img, DTYPE* tmp_a1, DTYPE* tmp_a2, int Nr, int Nc, int hlen, int level);
    void forward_swt_pass2(DTYPE* tmp_a1, DTYPE* tmp_a2, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
    void inverse_swt_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int hlen, int level);
    void inverse_swt_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int hlen, int level);


	aligned_data_vec_t c_kern_L;
	aligned_data_vec_t c_kern_H;
	aligned_data_vec_t c_kern_IL;
	aligned_data_vec_t c_kern_IH;
    aligned_data_vec_t tmp;
    w_info winfos;
};


#endif

