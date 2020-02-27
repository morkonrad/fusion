#ifndef NONSEPARABLE_H
#define NONSEPARABLE_H

#include "utils.h"
#include "filters.h"
#include "data_types.h"


class nonseparable_wavelet_transform{
	
	public:
	
	nonseparable_wavelet_transform(w_info winfo);
	~nonseparable_wavelet_transform();
	
    int set_filters(const char* wname, int do_swt);
    int set_filters_forward(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len);
    int set_filters_inverse(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len);
    
    int forward_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int inverse_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int forward_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
    int inverse_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs);
	
	private:

    void forward(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen);
    void inverse(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int Nr2, int Nc2, int hlen);
    void forward_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
    void inverse_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level);
    DTYPE* w_outer(DTYPE* a, DTYPE* b, int len);

	aligned_data_vec_t c_kern_LL;
	aligned_data_vec_t c_kern_LH;
	aligned_data_vec_t c_kern_HL;
	aligned_data_vec_t c_kern_HH;

	aligned_data_vec_t c_kern_ILL;
	aligned_data_vec_t c_kern_ILH;
	aligned_data_vec_t c_kern_IHL;
	aligned_data_vec_t c_kern_IHH;

    aligned_data_vec_t tmp;
    w_info winfos;
};


#endif
