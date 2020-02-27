#include "Lap_Pyr_Kernels.h"

__constant__ float *c_Kernel;

int checkdim(int dim, int dividor) {

	int result = dim / dividor;

	if (dim % dividor != 0) result++;

	return result;
}

__global__ void subtract(float *d_dst, float*d_src_1, float* d_src_2, int len) {


	int baseX = blockIdx.x * blockDim.x + threadIdx.x;

	if (baseX < len)
	{
		d_dst[baseX] = d_src_1[baseX] - d_src_2[baseX];
	}

}

void CU_subtract(float * d_dst, float * d_Src_1, float * d_Src_2, const int &len,cudaStream_t stream)
{

	dim3 threads(512, 1);
	dim3 blocks(checkdim(len, 512), 1);

    subtract <<<blocks, threads, 0, stream >>>(d_dst, d_Src_1, d_Src_2, len);
    cudaError_t err = cudaGetLastError();
    if(err!=0)exit(-1);

}

__global__ void add(float *d_dst, float*d_src_1, int len) {


	int baseX = blockIdx.x * blockDim.x + threadIdx.x;

	if (baseX < len)
	{
		d_dst[baseX] = d_dst[baseX] + d_src_1[baseX];
	}

}

void CU_add(float * d_dst, float * d_lap, const int &len) {

	dim3 threads(512, 1);
	dim3 blocks(checkdim(len, 512), 1);

    add <<<blocks, threads >> >(d_dst, d_lap, len);
    cudaError_t err = cudaGetLastError();
    if(err!=0)exit(-1);

}

__global__ void compare(float *d_ip_v, float *d_ip_ir, int len) {

	const int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < len)
	{
		d_ip_v[X] = (abs(d_ip_v[X]) > abs(d_ip_ir[X])) ? d_ip_v[X] : d_ip_ir[X];
	}

}

__global__ void average(float *d_ip_v, float *d_ip_ir, int app_len) {

	const int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < app_len)
	{
		d_ip_v[X] = (d_ip_v[X] + d_ip_ir[X]) / 2;
	}

}

void compare_Lp(float *d_ip_v, float *d_ip_ir, int app_len, int out_len,cudaStream_t stream1,cudaStream_t stream2) {

	int rem_len = out_len - app_len;

	dim3 blocks(checkdim(app_len, 1024), 1);
	dim3 blocks1(checkdim(rem_len, 1024), 1);
	dim3 threads(1024, 1);

	average << <blocks, threads, 0, stream1 >> > (d_ip_v, d_ip_ir, app_len);
    getLastCudaError("compare_Lp-avarage failed \n");

	compare << <blocks1, threads, 0, stream2 >> > (d_ip_v + app_len, d_ip_ir + app_len, rem_len);
    getLastCudaError("compare_Lp-compare failed \n");
}

void set_filter(float *d_Kernel)
{
	cudaMemcpyToSymbol(c_Kernel, &d_Kernel, sizeof(float*));
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionRowsKernel_v1(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int filter_Rad,
	int Halo_steps
)
{
	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - Halo_steps) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

	d_Src += baseY * imageW + baseX;
	d_Dst += baseY * imageW + baseX;

	//Load main data
	/*#pragma unroll

	for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS; i++)
	{
	s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
	}*/

	//Load left halo
    #pragma unroll
	for (int i = 0; i < Halo_steps; i++)
	{
		s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo and main data
    #pragma unroll
	for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS + Halo_steps; i++)
	{
		s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Compute and store results
	__syncthreads();
    #pragma unroll
	for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS; i++)
	{
		float sum = 0;
		if (baseX + i * ROWS_BLOCKDIM_X < imageW)
		{
            #pragma unroll
			for (int j = -filter_Rad; j <= filter_Rad; j++)
			{
				sum += c_Kernel[filter_Rad - j] * s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			}

			d_Dst[i * ROWS_BLOCKDIM_X] = sum;

		}
	}
}

void convolutionRowsGPU_v1(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int filter_Rad,
	int Halo_steps
)
{
	assert(ROWS_BLOCKDIM_X * Halo_steps >= filter_Rad);
	//assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	//assert(imageH % ROWS_BLOCKDIM_Y == 0);

	dim3 blocks(checkdim(imageW, (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)), checkdim(imageH, ROWS_BLOCKDIM_Y));
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	int s_data_size = (ROWS_BLOCKDIM_Y*((ROWS_RESULT_STEPS + (2 * Halo_steps)) * ROWS_BLOCKDIM_X)) * sizeof(float);

    convolutionRowsKernel_v1 <<<blocks, threads, s_data_size >>>(
		d_Dst,
		d_Src,
		imageW,
		filter_Rad,
		Halo_steps
		);
    //getLastCudaError("convolutionRowsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnsKernel_v1(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int pitch,
	int filter_Rad,
	int Halo_steps
)
{
	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - Halo_steps) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	/*	//Main data
	#pragma unroll

	for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS; i++)
	{
	s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS+2*Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
	}*/

	//Upper halo
    #pragma unroll
	for (int i = 0; i < Halo_steps; i++)
	{
		s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Lower halo + Main data
    #pragma unroll
	for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS + Halo_steps; i++)
	{
		s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Compute and store results
	__syncthreads();
    #pragma unroll
	for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS; i++)
	{
		float sum = 0;

        if (baseY + i * COLUMNS_BLOCKDIM_Y < imageH)
        {
            #pragma unroll
			for (int j = -filter_Rad; j <= filter_Rad; j++)
			{
				sum += c_Kernel[filter_Rad - j] * s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			}

			d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
		}
	}
}

void convolutionColumnsGPU_v1(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int filter_Rad,
	int Halo_steps
)
{
	assert(COLUMNS_BLOCKDIM_Y * Halo_steps >= filter_Rad);
	//assert(imageW % COLUMNS_BLOCKDIM_X == 0);
	//assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	dim3 blocks(checkdim(imageW, COLUMNS_BLOCKDIM_X), checkdim(imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
	dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	int s_data_size = sizeof(float) * (COLUMNS_BLOCKDIM_X*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) * COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel_v1 <<<blocks, threads, s_data_size >>>(
		d_Dst,
		d_Src,
		imageW,
		imageH,
		imageW,
		filter_Rad,
		Halo_steps
		);
    //getLastCudaError("convolutionColumnsKernel() execution failed\n");
}



__global__ void convolutionRowsKernel_down_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int n_imageW,
	int imageH,
	int filter_Rad,
	int Halo_steps
)
{
	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * 2 * ROWS_RESULT_STEPS - Halo_steps) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseX1 = (blockIdx.x * ROWS_RESULT_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    if (baseY < imageH)
    {
		d_Src += baseY * imageW + baseX;
		d_Dst += baseY * n_imageW + baseX1;

		//Load left halo
        #pragma unroll
		for (int i = 0; i < Halo_steps; ++i)
		{
			s_Data[(threadIdx.y*(2 * ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
		}

		//Load right halo and main data
        #pragma unroll
		for (int i = Halo_steps; i < Halo_steps + 2 * ROWS_RESULT_STEPS + Halo_steps; ++i)
		{
			s_Data[(threadIdx.y*(2 * ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
		}

		//Compute and store results
		__syncthreads();

        #pragma unroll
		for (int i = 0; i < ROWS_RESULT_STEPS; ++i)
		{
			float sum = 0;
			if (baseX1 + i * ROWS_BLOCKDIM_X < n_imageW)
			{
                #pragma unroll
				for (int j = -filter_Rad; j <= filter_Rad; ++j)
				{
					sum += c_Kernel[filter_Rad - j] * s_Data[(threadIdx.y*(2 * ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + (Halo_steps + 2 * i) * ROWS_BLOCKDIM_X + threadIdx.x * 2 + j];
				}

				d_Dst[i * ROWS_BLOCKDIM_X] = sum;

			}
		}
	}
}

void convolutionRowsGPU_down_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int n_imageW,
	int imageH,
	int filter_Rad,
	int Halo_steps,
    cudaStream_t stream
)
{
	assert(ROWS_BLOCKDIM_X * Halo_steps >= filter_Rad);

    dim3 blocks(checkdim(n_imageW, (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)), checkdim(imageH, ROWS_BLOCKDIM_Y));
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	int s_data_size = (ROWS_BLOCKDIM_Y*((2 * ROWS_RESULT_STEPS + (2 * Halo_steps)) * ROWS_BLOCKDIM_X)) * sizeof(float);

    convolutionRowsKernel_down_smp <<<blocks, threads, s_data_size, stream >>>(
		d_Dst,
		d_Src,
		imageW,
		n_imageW,
		imageH,
		filter_Rad,
		Halo_steps
		);
    ////getLastCudaError("convolutionRowsKernel() execution failed\n");
}


__global__ void convolutionColumnsKernel_down_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int n_imageH,
	int pitch,
	int filter_Rad,
	int Halo_steps
)
{
	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * 2 * COLUMNS_RESULT_STEPS - Halo_steps) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	const int baseY1 = (blockIdx.y * COLUMNS_RESULT_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

	if (baseX < imageW)
	{
		d_Src += baseY * pitch + baseX;
		d_Dst += baseY1 * pitch + baseX;

		//Upper halo
        #pragma unroll
		for (int i = 0; i < Halo_steps; i++)
		{
			s_Data[(threadIdx.x*(2 * COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
		}

		//Lower halo + Main data
        #pragma unroll
		for (int i = Halo_steps; i < Halo_steps + 2 * COLUMNS_RESULT_STEPS + Halo_steps; i++)
		{
			s_Data[(threadIdx.x*(2 * COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
		}

		//Compute and store results
		__syncthreads();
        #pragma unroll
		for (int i = 0; i < COLUMNS_RESULT_STEPS; ++i)
		{
			float sum = 0;

            if (baseY1 + i * COLUMNS_BLOCKDIM_Y < n_imageH)
            {
                #pragma unroll
				for (int j = -filter_Rad; j <= filter_Rad; ++j)
				{
					sum += c_Kernel[filter_Rad - j] * s_Data[(threadIdx.x*(2 * COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * COLUMNS_BLOCKDIM_Y + Halo_steps * COLUMNS_BLOCKDIM_Y + j];
				}

				d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
			}
		}
	}
}

void convolutionColumnsGPU_down_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int n_imageH,
	int filter_Rad,
	int Halo_steps,
    cudaStream_t stream
)
{
	assert(COLUMNS_BLOCKDIM_Y * Halo_steps >= filter_Rad);

	dim3 blocks(checkdim(imageW, COLUMNS_BLOCKDIM_X), checkdim(n_imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
	dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	int s_data_size = sizeof(float) * (COLUMNS_BLOCKDIM_X*(2 * COLUMNS_RESULT_STEPS + 2 * Halo_steps) * COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel_down_smp <<<blocks, threads, s_data_size, stream >>>(
		d_Dst,
		d_Src,
		imageW,
		imageH,
		n_imageH,
		imageW,
		filter_Rad,
		Halo_steps
		);
    ////getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

__global__ void convolutionRowsKernel_up_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int n_imageW,
	int imageH,
	int filter_Rad,
	int Halo_steps
)
{
	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - Halo_steps) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
	const int baseX1 = blockIdx.x * ROWS_RESULT_STEPS * 2 * ROWS_BLOCKDIM_X + 2 * threadIdx.x;

    if (baseY < imageH)
    {
		d_Src += baseY * imageW + baseX;
		d_Dst += baseY * n_imageW + baseX1;

		//Load left halo
        //#pragma unroll
		for (int i = 0; i < Halo_steps; ++i)
		{
			s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
		}

		//Load right halo and main data
        //#pragma unroll
		for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS + Halo_steps; ++i)
		{
			s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
		}

		//Compute and store results
		__syncthreads();

        //#pragma unroll
		for (int i = Halo_steps; i < Halo_steps + ROWS_RESULT_STEPS; ++i)
		{
			int pos_x = (baseX1 + 2 * (i - Halo_steps) * ROWS_BLOCKDIM_X);

			if (pos_x < n_imageW)
			{                
				float sum_1 = 0.0f, sum_2 = 0.0f;

                //#pragma unroll
				for (int l = -(filter_Rad / 2); l <= filter_Rad / 2; ++l)
				{
					int t = 2 * l;

					float temp = s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X + l];
					sum_1 += c_Kernel[filter_Rad + t] * temp *2.0f;
					sum_2 += c_Kernel[filter_Rad + t - 1] * temp *2.0f;

				}

				sum_2 += c_Kernel[2 * filter_Rad] * 2.0f * s_Data[(threadIdx.y*(ROWS_RESULT_STEPS + 2 * Halo_steps)*ROWS_BLOCKDIM_X) + threadIdx.x + i * ROWS_BLOCKDIM_X + filter_Rad / 2 + 1];

				d_Dst[2 * (i - Halo_steps)* ROWS_BLOCKDIM_X] = sum_1;
				if (pos_x + 1 < n_imageW) d_Dst[2 * (i - Halo_steps) * ROWS_BLOCKDIM_X + 1] = sum_2;
			}
		}
	}
}

void convolutionRowsGPU_up_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int n_imageW,
	int imageH,
	int filter_Rad,
	int Halo_steps,
    cudaStream_t stream
)
{
	assert(ROWS_BLOCKDIM_X * Halo_steps >= filter_Rad);
    //assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    //assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(checkdim(imageW, (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)), checkdim(imageH, ROWS_BLOCKDIM_Y));
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	int s_data_size = (ROWS_BLOCKDIM_Y*((ROWS_RESULT_STEPS + (2 * Halo_steps)) * ROWS_BLOCKDIM_X)) * sizeof(float);

    convolutionRowsKernel_up_smp <<<blocks, threads, s_data_size, stream >>>(
		d_Dst,
		d_Src,
		imageW,
		n_imageW,
		imageH,
		filter_Rad,
		Halo_steps
		);
    //getLastCudaError("convolutionRowsKernel() execution failed\n");
}

__global__ void convolutionColumnsKernel_up_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int n_imageH,
	int pitch,
	int filter_Rad,
	int Halo_steps
)
{
	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - Halo_steps) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

	if (baseX < imageW)
	{
		d_Src += baseY * pitch + baseX;
		d_Dst += 2 * baseY * pitch + baseX;

		//Upper halo
        //#pragma unroll
		for (int i = 0; i < Halo_steps; i++)
		{
			s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
		}

		//Lower halo + Main data
        //#pragma unroll
		for (int i = Halo_steps; i < Halo_steps + COLUMNS_RESULT_STEPS + Halo_steps; i++)
		{
			s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
		}

		//Compute and store results
		__syncthreads();
        //#pragma unroll
		for (int i = Halo_steps; i < COLUMNS_RESULT_STEPS + Halo_steps; ++i)
		{
			int Pos_y = 2 * baseY + (2 * i) * COLUMNS_BLOCKDIM_Y;

            if (Pos_y < n_imageH)
            {
				float sum_1 = 0.0f, sum_2 = 0.0f;

                //#pragma unroll
				for (int l = -(filter_Rad / 2); l <= filter_Rad / 2; ++l)
				{
					int t = 2 * l;

					float temp = s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y + l];

					sum_1 += c_Kernel[filter_Rad + t] * temp * 2.0f;
					sum_2 += c_Kernel[filter_Rad + t - 1] * temp * 2.0f;
				}

				sum_2 += c_Kernel[2 * filter_Rad] * 2.0f * s_Data[(threadIdx.x*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) *COLUMNS_BLOCKDIM_Y) + threadIdx.y + i * COLUMNS_BLOCKDIM_Y + filter_Rad / 2 + 1];

				d_Dst[2 * i * COLUMNS_BLOCKDIM_Y * pitch] = sum_1;
				if (Pos_y + 1 < n_imageH)d_Dst[2 * i * COLUMNS_BLOCKDIM_Y * pitch + pitch] = sum_2;

			}

		}
	}
}

void convolutionColumnsGPU_up_smp(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int n_imageH,
	int filter_Rad,
	int Halo_steps,
    cudaStream_t stream
)
{
	assert(COLUMNS_BLOCKDIM_Y * Halo_steps >= filter_Rad);
    //assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    //assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	dim3 blocks(checkdim(imageW, COLUMNS_BLOCKDIM_X), checkdim(n_imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
	dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	int s_data_size = sizeof(float) * (COLUMNS_BLOCKDIM_X*(COLUMNS_RESULT_STEPS + 2 * Halo_steps) * COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel_up_smp << <blocks, threads, s_data_size, stream >>>(
		d_Dst,
		d_Src,
		imageW,
		imageH,
		n_imageH,
		imageW,
		filter_Rad,
		Halo_steps
		);
    //getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

__global__ void d_transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

	if ((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

void transpose(float *d_src, float *d_dest, int width, int height,cudaStream_t stream)
{
	dim3 grid(checkdim(width, BLOCK_DIM), checkdim(height, BLOCK_DIM), 1);
	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose <<< grid, threads,0,stream >>>(d_dest, d_src, width, height);
    ////getLastCudaError("Kernel execution failed");
}

__global__ void
d_recursiveGaussian_down_smp_1(float *id, float *od, int w, int h)
{

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int cA_h = (int)ceil((double)h / 2.0);

	if (x >= w) return;

	id += x;    // advance pointers to correct column
	od += x;

	// forward pass
	float xp = 0;  // previous input
	float yp = 0;  // previous output
	float yb = 0;  // previous output by 2
#if CLAMP_TO_EDGE
	xp = *id;
	yb = c_Kernel[6] * xp;
	yp = yb;
#endif

	for (int y = 0; y < cA_h; y++)
	{
		float yc, xc = *id;
		yc = c_Kernel[0] * xc + c_Kernel[1] * xp - c_Kernel[4] * yp - c_Kernel[5] * yb; //yc = a0*xc + a1*xp - b1*yp - b2*yb;
		*od = yc;
		id += w;
		od += w;    // move to next Row
		xp = xc;
		yb = yp;
		yp = yc;

		if (2 * (y + 1) <= h) {

			yc, xc = *id;
			yc = c_Kernel[0] * xc + c_Kernel[1] * xp - c_Kernel[4] * yp - c_Kernel[5] * yb;
			//*od = yc;
			id += w; //to avoid accessing wrong values

					 //od += w;    // output dont move to next Row, downsampling by the factor of 2
			xp = xc;
			yb = yp;
			yp = yc;

		}
	}

	// reset pointers to point to last element in column
	id -= w;
	od -= w;

	// reverse pass
	// ensures response is symmetrical
	float xn = 0.0f;
	float xa = 0.0f;
	float yn = 0.0f;
	float ya = 0.0f;
#if CLAMP_TO_EDGE 
	xn = xa = *id;
	yn = c_Kernel[7] * xn;
	ya = yn;
#endif

	for (int y = cA_h - 1; y >= 0; y--)
	{

		if (2 * (y + 1) <= h) {

			float yc, xc = *id;
			yc = c_Kernel[2] * xn + c_Kernel[3] * xa - c_Kernel[4] * yn - c_Kernel[5] * ya;
			xa = xn;
			xn = xc;
			ya = yn;
			yn = yc;
			//*od = *od + yc;
			id -= w;
			//od -= w;  // move to previous row

		}

		float yc, xc = *id;
		yc = c_Kernel[2] * xn + c_Kernel[3] * xa - c_Kernel[4] * yn - c_Kernel[5] * ya;
		xa = xn;
		xn = xc;
		ya = yn;
		yn = yc;
		*od = *od + yc;
		id -= w;
		od -= w;
	}
}

void convolutionRecursive_down_smp_1(float * d_dst, float * d_src, float* d_buffer, int imageW, int imageH, int cA_h, int cA_w,cudaStream_t stream)
{
    d_recursiveGaussian_down_smp_1 <<< checkdim(imageW, BLOCK_DIM), BLOCK_DIM,0,stream >>>(d_src, d_buffer, imageW, imageH);
    ////getLastCudaError("Kernel execution failed");

    transpose(d_buffer, d_dst, imageW, cA_h,stream);
    ////getLastCudaError("transpose: Kernel execution failed");

    d_recursiveGaussian_down_smp_1 << < checkdim(cA_h, BLOCK_DIM), BLOCK_DIM,0,stream >> >(d_dst, d_buffer, cA_h, imageW);
    ////getLastCudaError("Kernel execution failed");

    transpose(d_buffer, d_dst, cA_h, cA_w,stream);
    ////getLastCudaError("transpose: Kernel execution failed");
}

__global__ void
d_recursiveGaussian_up_smp_1(float *id, float *od, int w, int h, int n_h)
{


	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;


	if (x >= w) return;

	id += x;    // advance pointers to correct column
	od += x;

	// forward pass
	float xp = 0;  // previous input
	float yp = 0;  // previous output
	float yb = 0;  // previous output by 2
#if CLAMP_TO_EDGE
	xp = *id;
	yb = c_Kernel[6] * xp;
	yp = yb;
#endif

	int f = 2;
	for (int y = 0; y < h; y++)
	{
		float yc, xc = *id;
		yc = c_Kernel[0] * xc*f + c_Kernel[1] * xp*f - c_Kernel[4] * yp - c_Kernel[5] * yb; //yc = a0*xc*f + a1*xp*f - b1*yp - b2*yb
		*od = yc;
		id += w;
		od += w;    // move to next Row
		xp = xc;
		yb = yp;
		yp = yc;

		if (2 * (y + 1) <= n_h) {

			//calculating for zero pixels inbetween
			yc, xc = 0;
			yc = c_Kernel[0] * xc*f + c_Kernel[1] * xp*f - c_Kernel[4] * yp - c_Kernel[5] * yb;//yc = a0*xc*f + a1*xp*f - b1*yp - b2*yb
			*od = yc;
			//id += w; //input point should not move to next Row, UPsampling by the factor of 2

			od += w;    //Moving to the next row 
			xp = xc;
			yb = yp;
			yp = yc;

		}
	}

	// reset pointers to point to last element in column
	id -= w;
	od -= w;

	// reverse pass
	// ensures response is symmetrical
	float xn = 0.0f;
	float xa = 0.0f;
	float yn = 0.0f;
	float ya = 0.0f;
#if CLAMP_TO_EDGE 

	if (n_h % 2 == 0) xn = xa = 0.0f;
	else xn = xa = *id;
	yn = c_Kernel[7] * xn;
	ya = yn;
#endif

	for (int y = h - 1; y >= 0; y--)
	{

		if (2 * (y + 1) <= n_h) {

			float yc, xc = 0;
			yc = c_Kernel[2] * xn*f + c_Kernel[3] * xa*f - c_Kernel[4] * yn - c_Kernel[5] * ya;  //yc = a2*xn*f + a3*xa*f - b1*yn - b2*ya
			xa = xn;
			xn = xc;
			ya = yn;
			yn = yc;
			*od = *od + yc;
			//id -= w;
			od -= w;  // move to previous row

		}

		float yc, xc = *id;
		yc = c_Kernel[2] * xn*f + c_Kernel[3] * xa*f - c_Kernel[4] * yn - c_Kernel[5] * ya;   //yc = a2*xn*f + a3*xa*f - b1*yn - b2*ya
		xa = xn;
		xn = xc;
		ya = yn;
		yn = yc;
		*od = *od + yc;
		id -= w;
		od -= w;
	}
}


void convolutionRecursive_up_smp_1(float * d_dst, float * d_src, float* d_buffer, int imageW, int imageH, int next_W, int next_H,cudaStream_t stream) {

    d_recursiveGaussian_up_smp_1 << < checkdim(imageW, BLOCK_DIM), BLOCK_DIM,0,stream >> >(d_src, d_buffer, imageW, imageH, next_H);
    ////getLastCudaError("Kernel execution failed");

    transpose(d_buffer, d_dst, imageW, next_H,stream);
    ////getLastCudaError("transpose: Kernel execution failed");

    d_recursiveGaussian_up_smp_1 << < checkdim(next_H, BLOCK_DIM), BLOCK_DIM,0,stream >> >(d_dst, d_buffer, next_H, imageW, next_W);
    ////getLastCudaError("Kernel execution failed");

    transpose(d_buffer, d_dst, next_H, next_W,stream);
    ////getLastCudaError("transpose: Kernel execution failed");
}
