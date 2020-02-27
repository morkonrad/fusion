#include "dwt_kernels.h"

__constant__ float *c_lpd;
__constant__ float *c_hpd;
__constant__ float *c_lpr;
__constant__ float *c_hpr;

int checkdim_dwt(int dim, int dividor) {

	int result = dim / dividor;

	if (dim % dividor != 0) result++;

	return result;
}

void set_filters(float* lpd, float* hpd, float* lpr, float* hpr)
{
	cudaMemcpyToSymbol(c_lpd, &lpd, sizeof(float*));
	cudaMemcpyToSymbol(c_hpd, &hpd, sizeof(float*));
	cudaMemcpyToSymbol(c_lpr, &lpr, sizeof(float*));
	cudaMemcpyToSymbol(c_hpr, &hpr, sizeof(float*));
    getLastCudaError("set_filters on cudaMemcpyToSymbol failed \n");
}

__global__ void dwt_per_X(float *d_ip, int rows, int cols, int cA_cols, int filt_len, int Halo_steps, float *d_cL, float *d_cH)
{
	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = ((blockIdx.x * 2 * X_RESULT_STEPS) - Halo_steps) * X_BLOCKDIM_X + threadIdx.x;
	const int baseX1 = (blockIdx.x * X_RESULT_STEPS) * X_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * X_BLOCKDIM_Y + threadIdx.y;

	if (baseY < rows) {

		d_ip += baseY * cols + baseX;
		d_cL += baseY * cA_cols + baseX1;
		d_cH += baseY * cA_cols + baseX1;

		//Loading data to shared memory
		if (cols % 2 == 1) {
			//Load Left Halo
#pragma unroll
			for (int i = 0; i < Halo_steps; i++)
			{
				if (baseX + i * X_BLOCKDIM_X == -1) s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = d_ip[cols - 1];

				else s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = (baseX + i * X_BLOCKDIM_X >= 0) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X + cols + 1];
			}

			// main data and Load right halo
#pragma unroll

			for (int i = Halo_steps; i < Halo_steps + 2 * X_RESULT_STEPS + Halo_steps; i++)
			{
				if (baseX + i * X_BLOCKDIM_X == cols) s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = d_ip[i * X_BLOCKDIM_X - 1];

				else s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = ((baseX + i * X_BLOCKDIM_X) < cols) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X - cols - 1];
			}

			//Compute and store results
			__syncthreads();
		}

		else
		{
#pragma unroll
			for (int i = 0; i < Halo_steps; i++)
			{
				s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = (baseX + i * X_BLOCKDIM_X >= 0) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X + cols];
			}

			// main data and Load right halo
#pragma unroll

			for (int i = Halo_steps; i < Halo_steps + 2 * X_RESULT_STEPS + Halo_steps; i++)
			{
				s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = ((baseX + i * X_BLOCKDIM_X) < cols) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X - cols];
			}

			//Compute and store results
			__syncthreads();
		}

#pragma unroll

		for (int i = 0; i < X_RESULT_STEPS; i++)
		{
			if ((baseX1 + i * X_BLOCKDIM_X < cA_cols))
			{
				float sum_cL = 0, sum_cH = 0;

				int l2 = filt_len / 2;

				for (int l = 0; l < filt_len; ++l)
				{
					sum_cL += c_lpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l]; //l2-l is to select the right center pixels with odd and even sized filters
					sum_cH += c_hpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l];
				}
				d_cL[i * X_BLOCKDIM_X] = sum_cL;
				d_cH[i * X_BLOCKDIM_X] = sum_cH;
			}
		}
	}
}

__global__ void dwt_per_X_O(float *d_ip, int rows, int cols, int cA_cols, int filt_len, int Halo_steps, float *d_cL, float *d_cH)
{
	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * 2 * X_RESULT_STEPS - Halo_steps) * X_BLOCKDIM_X + threadIdx.x;
	const int baseX1 = (blockIdx.x * X_RESULT_STEPS) * X_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * X_BLOCKDIM_Y + threadIdx.y;

	if (baseY < rows) {

		d_ip += baseY * cols + baseX;
		d_cL += baseY * cA_cols + baseX1;
		d_cH += baseY * cA_cols + baseX1;

		//Loading data to shared memory

		//Load Left Halo
#pragma unroll
		for (int i = 0; i < Halo_steps; i++)
		{
			if (baseX + i * X_BLOCKDIM_X == -1) s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = d_ip[cols - 1];

			else s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = (baseX + i * X_BLOCKDIM_X >= 0) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X + cols + 1];
		}

		// main data and Load right halo
#pragma unroll

		for (int i = Halo_steps; i < Halo_steps + 2 * X_RESULT_STEPS + Halo_steps; i++)
		{
			if (baseX + i * X_BLOCKDIM_X == cols) s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = d_ip[i * X_BLOCKDIM_X - 1];

			else s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = ((baseX + i * X_BLOCKDIM_X) < cols) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X - cols - 1];
		}

		//Compute and store results
		__syncthreads();


#pragma unroll

		for (int i = 0; i < X_RESULT_STEPS; i++)
		{
			if ((baseX1 + i * X_BLOCKDIM_X < cA_cols))
			{
				float sum_cL = 0, sum_cH = 0;

				int l2 = filt_len / 2;

				for (int l = 0; l < filt_len; ++l)
				{
					sum_cL += c_lpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l]; //l2-l is to select the right center pixels with odd and even sized filters
					sum_cH += c_hpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l];
				}
				d_cL[i * X_BLOCKDIM_X] = sum_cL;
				d_cH[i * X_BLOCKDIM_X] = sum_cH;
			}
		}
	}
}

__global__ void dwt_per_X_E(float *d_ip, int rows, int cols, int cA_cols, int filt_len, int Halo_steps, float *d_cL, float *d_cH)
{
	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * 2 * X_RESULT_STEPS - Halo_steps) * X_BLOCKDIM_X + threadIdx.x;
	const int baseX1 = (blockIdx.x * X_RESULT_STEPS) * X_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * X_BLOCKDIM_Y + threadIdx.y;

	if (baseY < rows) {

		d_ip += baseY * cols + baseX;
		d_cL += baseY * cA_cols + baseX1;
		d_cH += baseY * cA_cols + baseX1;

		//Loading data to shared memory
#pragma unroll
		for (int i = 0; i < Halo_steps; i++)
		{
			s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = (baseX + i * X_BLOCKDIM_X >= 0) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X + cols];
		}

		// main data and Load right halo
#pragma unroll

		for (int i = Halo_steps; i < Halo_steps + 2 * X_RESULT_STEPS + Halo_steps; i++)
		{
			s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x + i * X_BLOCKDIM_X] = ((baseX + i * X_BLOCKDIM_X) < cols) ? d_ip[i * X_BLOCKDIM_X] : d_ip[i * X_BLOCKDIM_X - cols];
		}

		//Compute and store results
		__syncthreads();

#pragma unroll

		for (int i = 0; i < X_RESULT_STEPS; i++)
		{
			if ((baseX1 + i * X_BLOCKDIM_X < cA_cols))
			{
				float sum_cL = 0, sum_cH = 0;

				int l2 = filt_len / 2;
#pragma unroll
				for (int l = 0; l < filt_len; ++l)
				{
					sum_cL += c_lpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l]; //l2-l is to select the right center pixels with odd and even sized filters
					sum_cH += c_hpd[l] * s_Data[(threadIdx.y*(2 * X_RESULT_STEPS + 2 * Halo_steps)*X_BLOCKDIM_X) + threadIdx.x * 2 + Halo_steps*X_BLOCKDIM_X + 2 * i * X_BLOCKDIM_X + l2 - l];
				}
				d_cL[i * X_BLOCKDIM_X] = sum_cL;
				d_cH[i * X_BLOCKDIM_X] = sum_cH;
			}
		}
	}
}

void DWT_X_GPU(float *d_ip, const int &rows, const int &cols, const int &cA_cols, const int &f_len, float *d_cL, float *d_cH, const int &halo, cudaStream_t stream) {

    dim3 blocks(checkdim_dwt(cA_cols, X_RESULT_STEPS*X_BLOCKDIM_X), checkdim_dwt(rows, X_BLOCKDIM_Y));
	dim3 threads(X_BLOCKDIM_X, X_BLOCKDIM_Y);

	int s_data_size = (X_BLOCKDIM_Y*((2 * X_RESULT_STEPS + (2 * halo)) * X_BLOCKDIM_X)) * sizeof(float); //2*X_RESULT_STEPS. Since we have created less threads correspond to cA_cols 

																										 //if (cols%2 == 0) dwt_per_X_E << < blocks, threads, s_data_size >> >(d_ip, rows, cols, cA_cols, f_len, halo_steps, d_cL, d_cH);
	if (cols % 2 == 0) dwt_per_X_E << < blocks, threads, s_data_size, stream >> >(d_ip, rows, cols, cA_cols, f_len, halo, d_cL, d_cH);
	else dwt_per_X_O << < blocks, threads, s_data_size, stream >> >(d_ip, rows, cols, cA_cols, f_len, halo, d_cL, d_cH);


    getLastCudaError("DWT_X_GPU convolution kernel failed \n");

}

__global__ void dwt_per_Y(float *d_ip, int rows, int cols, int cA_rows, int filt_len, int Halo_steps, float *d_cL, float *d_cH) {

	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * Y_BLOCKDIM_X + threadIdx.x;
	const int baseY = ((blockIdx.y * 2 * Y_RESULT_STEPS) - Halo_steps) * Y_BLOCKDIM_Y + threadIdx.y;
	const int baseY1 = (blockIdx.y * Y_RESULT_STEPS) * Y_BLOCKDIM_Y + threadIdx.y;

	if (baseX < cols)
	{
		d_ip += baseY * cols + baseX;
		d_cL += baseY1 * cols + baseX;
		d_cH += baseY1 * cols + baseX;

		//Loading data to shared memory
		if (rows % 2 == 1)
		{
			//Upper halo
#pragma unroll

			for (int i = 0; i < Halo_steps; i++)
			{
				if (baseY + i * Y_BLOCKDIM_Y == -1) s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = d_ip[(rows - 1) * cols];

				else s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY >= -i * Y_BLOCKDIM_Y) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) + ((rows + 1)*cols)];
			}

			//Lower halo + Main data
#pragma unroll
			for (int i = Halo_steps; i < Halo_steps + 2 * Y_RESULT_STEPS + Halo_steps; i++)
			{
				if (baseY + i * Y_BLOCKDIM_Y == rows) s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = d_ip[(i * Y_BLOCKDIM_Y * (cols - 1))];

				else s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (rows - baseY > i * Y_BLOCKDIM_Y) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) - ((rows + 1)*cols)];
			}

			__syncthreads();
		}

		else
		{
			//Upper halo
#pragma unroll

			for (int i = 0; i < Halo_steps; i++)
			{
				s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY >= -i * Y_BLOCKDIM_Y) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) + (rows*cols)];
			}

			//Lower halo + Main data
#pragma unroll
			for (int i = Halo_steps; i < Halo_steps + 2 * Y_RESULT_STEPS + Halo_steps; i++)
			{
				s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (rows - baseY > i * Y_BLOCKDIM_Y) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) - (rows*cols)];
			}

			__syncthreads();
		}

		//Compute and store results
#pragma unroll
		for (int i = 0; i < Y_RESULT_STEPS; i++)
		{
			if ((baseY1 + i * Y_BLOCKDIM_Y < cA_rows)) {
				int l2 = filt_len / 2;

				float sum_cL = 0, sum_cH = 0;

				for (int l = 0; l < filt_len; ++l)
				{
					sum_cL += c_lpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
					sum_cH += c_hpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
				}

				d_cL[i * Y_BLOCKDIM_Y * cols] = sum_cL;
				d_cH[i * Y_BLOCKDIM_Y * cols] = sum_cH;
			}
		}
	}
}

__global__ void dwt_per_Y_E(float *d_ip, int rows, int cols, int cA_rows, int filt_len, int Halo_steps, float *d_cL, float *d_cH) {
	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * Y_BLOCKDIM_X + threadIdx.x;
	const int baseY = ((blockIdx.y * 2 * Y_RESULT_STEPS) - Halo_steps) * Y_BLOCKDIM_Y + threadIdx.y;
	const int baseY1 = (blockIdx.y * Y_RESULT_STEPS) * Y_BLOCKDIM_Y + threadIdx.y;

	if (baseX < cols)
	{
		d_ip += baseY * cols + baseX;
		d_cL += baseY1 * cols + baseX;
		d_cH += baseY1 * cols + baseX;

		//Loading data to shared memory
		//Upper halo
#pragma unroll

		for (int i = 0; i < Halo_steps; i++)
		{
			s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY + i * Y_BLOCKDIM_Y >= 0) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) + (rows*cols)];
		}

		//Lower halo + Main data
#pragma unroll
		for (int i = Halo_steps; i < Halo_steps + 2 * Y_RESULT_STEPS + Halo_steps; i++)
		{
			s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY + i * Y_BLOCKDIM_Y < rows) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) - (rows*cols)];
		}

		__syncthreads();

		//Compute and store results
#pragma unroll
		for (int i = 0; i < Y_RESULT_STEPS; i++)
		{
			if ((baseY1 + i * Y_BLOCKDIM_Y < cA_rows)) {
				int l2 = filt_len / 2;

				float sum_cL = 0, sum_cH = 0;

				for (int l = 0; l < filt_len; ++l)
				{
					sum_cL += c_lpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
					sum_cH += c_hpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
				}

				d_cL[i * Y_BLOCKDIM_Y * cols] = sum_cL;
				d_cH[i * Y_BLOCKDIM_Y * cols] = sum_cH;
			}
		}
	}

}

__global__ void dwt_per_Y_O(float *d_ip, int rows, int cols, int cA_rows, int filt_len, int Halo_steps, float *d_cL, float *d_cH)
{
	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * Y_BLOCKDIM_X + threadIdx.x;
	const int baseY = ((blockIdx.y * 2 * Y_RESULT_STEPS) - Halo_steps) * Y_BLOCKDIM_Y + threadIdx.y;
	const int baseY1 = (blockIdx.y * Y_RESULT_STEPS) * Y_BLOCKDIM_Y + threadIdx.y;

	if (baseX < cols)
	{
		d_ip += baseY * cols + baseX;
		d_cL += baseY1 * cols + baseX;
		d_cH += baseY1 * cols + baseX;

		//Loading data to shared memory
		//Upper halo
#pragma unroll

		for (int i = 0; i < Halo_steps; i++)
		{
			if (baseY + i * Y_BLOCKDIM_Y == -1) s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = d_ip[(rows - 1) * cols];

			else s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY + i * Y_BLOCKDIM_Y >= 0) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) + ((rows + 1)*cols)];
		}

		//Lower halo + Main data
#pragma unroll
		for (int i = Halo_steps; i < Halo_steps + 2 * Y_RESULT_STEPS + Halo_steps; i++)
		{
			if (baseY + i * Y_BLOCKDIM_Y == rows) s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = d_ip[(i * Y_BLOCKDIM_Y * (cols - 1))];

			else s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + threadIdx.y + i * Y_BLOCKDIM_Y] = (baseY + i * Y_BLOCKDIM_Y < rows) ? d_ip[i * Y_BLOCKDIM_Y * cols] : d_ip[(i * Y_BLOCKDIM_Y * cols) - ((rows + 1)*cols)];
		}

		__syncthreads();
		//Compute and store results
#pragma unroll
		for (int i = 0; i < Y_RESULT_STEPS; i++)
		{
			if ((baseY1 + i * Y_BLOCKDIM_Y < cA_rows)) {
				int l2 = filt_len / 2;

				float sum_cL = 0, sum_cH = 0;

				for (int l = 0; l < filt_len; ++l)
				{
					sum_cL += c_lpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
					sum_cH += c_hpd[l] * s_Data[(threadIdx.x*(2 * Y_RESULT_STEPS + 2 * Halo_steps) *Y_BLOCKDIM_Y) + 2 * threadIdx.y + 2 * i * Y_BLOCKDIM_Y + Halo_steps * Y_BLOCKDIM_Y + l2 - l];
				}

				d_cL[i * Y_BLOCKDIM_Y * cols] = sum_cL;
				d_cH[i * Y_BLOCKDIM_Y * cols] = sum_cH;
			}
		}
	}
}

void DWT_Y_GPU(float *d_ip, const int &rows, const int &cols, const int &cA_rows, const int &f_len, float *d_cL, float *d_cH, const int &halo, cudaStream_t stream) {

    dim3 blocks(checkdim_dwt(cols, Y_BLOCKDIM_X), checkdim_dwt(cA_rows, (Y_RESULT_STEPS * Y_BLOCKDIM_Y)));
	dim3 threads(Y_BLOCKDIM_X, Y_BLOCKDIM_Y);

	int s_data_size = (Y_BLOCKDIM_Y*((2 * Y_RESULT_STEPS) + (2 * halo)) * Y_BLOCKDIM_X) * sizeof(float); //2*Y_RESULT_STEPS. Since we have created less threads correspond to cA_cols 

	dwt_per_Y << < blocks, threads, s_data_size, stream >> >(d_ip, rows, cols, cA_rows, f_len, halo, d_cL, d_cH);
	//else dwt_per_Y_O << < blocks, threads, s_data_size, stream>> >(d_ip, rows, cols, cA_rows, f_len, halo, d_cL, d_cH);


    getLastCudaError("DWT_Y_GPU convolution kernel failed \n");

}

__global__ void idwt_per_X_1(float *d_dst, float *src_A, float *src_D, int rows, int cols, int next_cols, int filt_len, int halo) {

	extern __shared__ float s_Data[];

	//Offset to the left halo edge
	const int baseX = ((blockIdx.x * I_X_RESULT_STEPS) - halo) * I_X_BLOCKDIM_X + threadIdx.x; // even if the last pixel of cols+l2-1 takes a new block, rows can be maintained
	const int baseY = blockIdx.y * I_X_BLOCKDIM_Y + threadIdx.y;

	const int baseX1 = blockIdx.x * I_X_RESULT_STEPS * 2 * I_X_BLOCKDIM_X + 2 * threadIdx.x;

	if (baseY < rows) {

		src_A += baseY * cols + baseX;
		src_D += baseY * cols + baseX;
		d_dst += baseY * next_cols + baseX1;//To compensate the halo

		int l2 = filt_len / 2;

		//Loading data to shared memory
#pragma unroll
		for (int i = 0; i < halo; i++)
		{
			s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = (baseX + i * I_X_BLOCKDIM_X >= 0) ? src_A[i * I_X_BLOCKDIM_X] : src_A[i * I_X_BLOCKDIM_X + cols];
			s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = (baseX + i * I_X_BLOCKDIM_X >= 0) ? src_D[i * I_X_BLOCKDIM_X] : src_D[i * I_X_BLOCKDIM_X + cols];
		}

		// main data and Load right halo
#pragma unroll

		for (int i = halo; i < halo + I_X_RESULT_STEPS + halo; i++)
		{

			s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = ((baseX + i * I_X_BLOCKDIM_X) < cols) ? src_A[i * I_X_BLOCKDIM_X] : src_A[i * I_X_BLOCKDIM_X - cols];
			s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X] = ((baseX + i * I_X_BLOCKDIM_X) < cols) ? src_D[i * I_X_BLOCKDIM_X] : src_D[i * I_X_BLOCKDIM_X - cols];

		}

		//Compute and store results
		__syncthreads();

#pragma unroll
		for (int i = halo; i < halo + I_X_RESULT_STEPS; i++)
		{
			int pos_x = (baseX1 + 2 * (i - halo) * I_X_BLOCKDIM_X);

			if ((pos_x + 1) < (2 * cols + filt_len - 2)) {

				float temp_1 = 0, temp_2 = 0;

				for (int l = 0; l < l2; ++l)
				{
					int t = 2 * l;

					temp_1 += c_lpr[t] * s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l]
						+ c_hpr[t] * s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l];
					temp_2 += c_lpr[t + 1] * s_Data[(threadIdx.y*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l]
						+ c_hpr[t + 1] * s_Data[((threadIdx.y + I_X_BLOCKDIM_Y)*(I_X_RESULT_STEPS + 2 * halo)*I_X_BLOCKDIM_X) + threadIdx.x + i * I_X_BLOCKDIM_X - l];
				}

				if ((pos_x >= l2 - 1) && (pos_x < (next_cols + l2 - 1))) d_dst[2 * (i - halo) * I_X_BLOCKDIM_X - l2 + 1] = temp_1;
				if ((pos_x + 1 >= l2 - 1) && (pos_x + 1 < (next_cols + l2 - 1))) d_dst[2 * (i - halo) * I_X_BLOCKDIM_X - l2 + 2] = temp_2;
			}

		}
	}
}

__global__ void idwt_per_Y_1(float *d_dst, float *src_A, float *src_D, int rows, int cols, int next_rows, int filt_len, int halo) {

	extern __shared__ float s_Data[];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * I_Y_BLOCKDIM_X + threadIdx.x;
	const int baseY = ((blockIdx.y * I_Y_RESULT_STEPS) - halo) * I_Y_BLOCKDIM_Y + threadIdx.y;

	int l2 = filt_len / 2;

	if (baseX < cols)
	{
		src_A += baseY * cols + baseX;
		src_D += baseY * cols + baseX;
		d_dst += (2 * baseY - l2 + 1) * cols + baseX;

		//Loading data to shared memory
		//Upper halo
#pragma unroll

		for (int i = 0; i < halo; i++)
		{
			s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y >= 0) ? src_A[i * I_Y_BLOCKDIM_Y * cols] : src_A[(i * I_Y_BLOCKDIM_Y * cols) + (rows*cols)];
			s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y >= 0) ? src_D[i * I_Y_BLOCKDIM_Y * cols] : src_D[(i * I_Y_BLOCKDIM_Y * cols) + (rows*cols)];
		}

		//Lower halo + Main data
#pragma unroll
		for (int i = halo; i < halo + I_Y_RESULT_STEPS + halo; i++)
		{
			s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y < rows) ? src_A[i * I_Y_BLOCKDIM_Y * cols] : src_A[(i * I_Y_BLOCKDIM_Y * cols) - (rows*cols)];
			s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y] = (baseY + i * I_Y_BLOCKDIM_Y < rows) ? src_D[i * I_Y_BLOCKDIM_Y * cols] : src_D[(i * I_Y_BLOCKDIM_Y * cols) - (rows*cols)];
		}

		__syncthreads();
#pragma unroll
		for (int i = halo; i < I_Y_RESULT_STEPS + halo; i++)
		{
			int pos_y = 2 * baseY + 2 * i * I_Y_BLOCKDIM_Y;

			if (pos_y + 1 < (2 * rows + filt_len - 2)) {

				float temp_1 = 0, temp_2 = 0;


				for (int l = 0; l < l2; ++l)
				{
					int t = 2 * l;

					temp_1 += c_lpr[t] * s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i* I_Y_BLOCKDIM_Y - l]
						+ c_hpr[t] * s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y - l];
					temp_2 += c_lpr[t + 1] * s_Data[(threadIdx.x*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y - l]
						+ c_hpr[t + 1] * s_Data[((threadIdx.x + I_Y_BLOCKDIM_X)*(I_Y_RESULT_STEPS + 2 * halo) *I_Y_BLOCKDIM_Y) + threadIdx.y + i * I_Y_BLOCKDIM_Y - l];
				}

				if ((pos_y >= l2 - 1) && (pos_y < next_rows + l2 - 1)) d_dst[2 * i * I_Y_BLOCKDIM_Y * cols] = temp_1;
				if ((pos_y + 1 >= l2 - 1) && (pos_y + 1 < next_rows + l2 - 1)) d_dst[(2 * i * I_Y_BLOCKDIM_Y + 1) * cols] = temp_2;
			}
		}

	}
}

void IDWT_X_GPU_1(float *d_dst, float*d_src_A, float *d_src_D, const int &rows, const int &cols, const int &next_cols, const int &filt_len, cudaStream_t stream) {

	int l2 = filt_len / 2;
	int halo = 1;

    dim3 blocks(checkdim_dwt((cols + l2 - 1), I_X_BLOCKDIM_X*I_X_RESULT_STEPS), checkdim_dwt(rows, I_X_BLOCKDIM_Y));
	dim3 threads(I_X_BLOCKDIM_X, I_X_BLOCKDIM_Y);

	int s_data_size = (I_X_BLOCKDIM_X * ((I_X_RESULT_STEPS)+(2 * halo)) * 2 * I_X_BLOCKDIM_Y) * sizeof(float);  //2*I_X_BLOCKDIM_Y is because we'll be loading data of both cL and cH

	idwt_per_X_1 << <blocks, threads, s_data_size, stream >> > (d_dst, d_src_A, d_src_D, rows, cols, next_cols, filt_len, halo);

    getLastCudaError("IDWT_X_GPU convolution kernel failed \n");

}

void IDWT_Y_GPU_1(float *d_dst, float*d_src_A, float *d_src_D, const int &rows, const int &cols, const int &next_rows, const int &filt_len, cudaStream_t stream) {

	int l2 = filt_len / 2;
	int halo = 1;

    dim3 blocks(checkdim_dwt(cols, I_Y_BLOCKDIM_X), checkdim_dwt((rows + l2 - 1), I_Y_BLOCKDIM_Y*I_Y_RESULT_STEPS));
	dim3 threads(I_Y_BLOCKDIM_X, I_Y_BLOCKDIM_Y);

	int s_data_size = (2 * I_Y_BLOCKDIM_X * I_Y_BLOCKDIM_Y * ((I_Y_RESULT_STEPS)+(2 * halo))) * sizeof(float);  //2 * I_Y_BLOCKDIM_X  is because we'll be loading data of both cL and cH 

	idwt_per_Y_1 << <blocks, threads, s_data_size, stream >> > (d_dst, d_src_A, d_src_D, rows, cols, next_rows, filt_len, halo);

    getLastCudaError("IDWT_Y_GPU convolution kernel failed \n");

}

__global__ void dwt_compare(float *d_ip_v, float *d_ip_ir, int len) {

	const int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < len)
	{
		d_ip_v[X] = (abs(d_ip_v[X]) > abs(d_ip_ir[X])) ? d_ip_v[X] : d_ip_ir[X];
	}

}

__global__ void dwt_average(float *d_ip_v, float *d_ip_ir, int app_len) {

	const int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < app_len)
	{
		d_ip_v[X] = (d_ip_v[X] + d_ip_ir[X]) / 2;
	}

}

void DWT_compare(float *d_ip_v, float *d_ip_ir, int app_len, int out_len, cudaStream_t stream1, cudaStream_t stream2)
{

    dim3 blocks(checkdim_dwt(app_len, 256), 1);
    dim3 blocks1(checkdim_dwt((out_len - app_len), 256), 1);
	dim3 threads(256, 1);

    dwt_average << <blocks, threads, NULL, stream1 >> > (d_ip_v, d_ip_ir, app_len);
    dwt_compare << <blocks1, threads, NULL, stream2 >> > (d_ip_v + app_len, d_ip_ir + app_len, (out_len - app_len));

    getLastCudaError("DWT_compare failed \n");
}
