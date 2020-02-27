#pragma once
#include "cuda_runtime_api.h"
#include "common.h"

namespace DWT_LIFT
{
	class dwt_lift
	{
		public:		
		dwt_lift();
		~dwt_lift();

        cudaError_t
        calc_forward(int DWT_Levels, int cols, int rows, DATATYPE* dev_in_image, DATATYPE* dev_out_image,cudaStream_t& stream)const;

        cudaError_t
        calc_inverse(int DWT_Levels, int cols, int rows, DATATYPE* dev_in_image, DATATYPE* dev_out_image,cudaStream_t& stream)const;
			
		private:

    };





}
