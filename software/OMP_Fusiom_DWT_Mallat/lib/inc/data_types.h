#ifndef _DATA_TYPES_H
#define _DATA_TYPES_H
#include <vector>
#include "aligned_allocator.h"

 using aligned_data_vec_t = std::vector<DTYPE,aligned_allocator<DTYPE,16>>;
 
 using wavelet_coeff_t = std::vector<aligned_data_vec_t>;
 
#endif
