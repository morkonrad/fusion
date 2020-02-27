#include "fusion_rule.h"


//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

__global__ void fuse_sum(
        const DATATYPE* device_image_vis,
        const DATATYPE* device_image_ir,
        DATATYPE* device_image_fused,
        int width)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int pix = IMAD(y,width,x);

    const DATATYPE vis = device_image_vis[pix];
    const DATATYPE ir = device_image_ir[pix];
    const DATATYPE fused = vis+ir;
    device_image_fused[pix]= fused;
}

__global__ void fuse_max(
        const DATATYPE* device_image_vis,
        const DATATYPE* device_image_ir,
        DATATYPE* device_image_fused,
        int width)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int pix = IMAD(y,width,x);

    const DATATYPE vis = device_image_vis[pix];
    const DATATYPE ir = device_image_ir[pix];
    device_image_fused[pix]= vis>ir?vis:ir;
}

__global__ void fuse_maxabs(
        const DATATYPE* device_image_vis,
        const DATATYPE* device_image_ir,
        DATATYPE* device_image_fused,
        int width)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int pix = IMAD(y,width,x);

    const DATATYPE vis = fabsf(device_image_vis[pix]);
    const DATATYPE ir = fabsf(device_image_ir[pix]);
    const DATATYPE fused = max(vis,ir);
    device_image_fused[pix]= fused;
}

__global__ void fuse_sumabs(
        const DATATYPE* device_image_vis,
        const DATATYPE* device_image_ir,
        DATATYPE* device_image_fused,
        int width)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int pix = IMAD(y,width,x);

    const DATATYPE vis = fabsf(device_image_vis[pix]);
    const DATATYPE ir = fabsf(device_image_ir[pix]);
    const DATATYPE fused = vis+ir;
    device_image_fused[pix]= fused;
}

__global__ void fuse_avg(
        const DATATYPE* device_image_vis,
        const DATATYPE* device_image_ir,
        DATATYPE* device_image_fused,
        int width)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int pix = IMAD(y,width,x);

    const DATATYPE vis = device_image_vis[pix];
    const DATATYPE ir = device_image_ir[pix];
    const DATATYPE fused = vis+ir;
    device_image_fused[pix] = fused/2;
}


namespace FUSE_RULE
{
cudaError_t calc_fuse_sum(const DATATYPE* dev_in_vis,
                          const DATATYPE* dev_in_ir,
                          DATATYPE* dev_out_fused,
                          const std::uint32_t& rows,
                          const std::uint32_t& cols,
                          cudaStream_t& stream)
{
    if(dev_in_ir==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_in_vis==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_out_fused==nullptr)return cudaError_t::cudaErrorAssert;
    if((rows*cols)==0)return cudaError_t::cudaErrorAssert;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(iDivUp(cols, threadsPerBlock.x), iDivUp(rows,threadsPerBlock.y));

    fuse_sum<<<numBlocks,threadsPerBlock,0,stream>>>(dev_in_vis,dev_in_ir,dev_out_fused,cols);

    return cudaError_t::cudaSuccess;
}
cudaError_t calc_fuse_maxabs(const DATATYPE* dev_in_vis,
                             const DATATYPE* dev_in_ir,
                             DATATYPE* dev_out_fused,
                             const std::uint32_t& rows,
                             const std::uint32_t& cols,
                             cudaStream_t& stream)
{
    if(dev_in_ir==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_in_vis==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_out_fused==nullptr)return cudaError_t::cudaErrorAssert;
    if((rows*cols)==0)return cudaError_t::cudaErrorAssert;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(iDivUp(cols, threadsPerBlock.x), iDivUp(rows,threadsPerBlock.y));

    fuse_maxabs<<<numBlocks,threadsPerBlock,0,stream>>>(dev_in_vis,dev_in_ir,dev_out_fused,cols);

    return cudaError_t::cudaSuccess;
}

cudaError_t calc_fuse_avg(const DATATYPE* dev_in_vis,
                          const DATATYPE* dev_in_ir,
                          DATATYPE* dev_out_fused,
                          const std::uint32_t& rows,
                          const std::uint32_t& cols,
                          cudaStream_t& stream)
{
    if(dev_in_ir==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_in_vis==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_out_fused==nullptr)return cudaError_t::cudaErrorAssert;
    if((rows*cols)==0)return cudaError_t::cudaErrorAssert;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(iDivUp(cols, threadsPerBlock.x), iDivUp(rows,threadsPerBlock.y));

    fuse_avg<<<numBlocks,threadsPerBlock,0,stream>>>(dev_in_vis,dev_in_ir,dev_out_fused,cols);

    return cudaError_t::cudaSuccess;
}


cudaError_t calc_fuse_sum_abs(const DATATYPE* dev_in_vis,
                              const DATATYPE* dev_in_ir,
                              DATATYPE* dev_out_fused,
                              const std::uint32_t& rows,
                              const std::uint32_t& cols,
                              cudaStream_t& stream)
{
    if(dev_in_ir==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_in_vis==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_out_fused==nullptr)return cudaError_t::cudaErrorAssert;
    if((rows*cols)==0)return cudaError_t::cudaErrorAssert;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(iDivUp(cols, threadsPerBlock.x), iDivUp(rows,threadsPerBlock.y));

    fuse_sumabs<<<numBlocks,threadsPerBlock,0,stream>>>(dev_in_vis,dev_in_ir,dev_out_fused,cols);

    return cudaError_t::cudaSuccess;
}

cudaError_t calc_fuse_max(const DATATYPE* dev_in_vis,
                          const DATATYPE* dev_in_ir,
                          DATATYPE* dev_out_fused,
                          const std::uint32_t& rows,
                          const std::uint32_t& cols,
                          cudaStream_t& stream)
{
    if(dev_in_ir==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_in_vis==nullptr)return cudaError_t::cudaErrorAssert;
    if(dev_out_fused==nullptr)return cudaError_t::cudaErrorAssert;
    if((rows*cols)==0)return cudaError_t::cudaErrorAssert;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(iDivUp(cols, threadsPerBlock.x), iDivUp(rows,threadsPerBlock.y));

    fuse_max<<<numBlocks,threadsPerBlock,0,stream>>>(dev_in_vis,dev_in_ir,dev_out_fused,cols);

    return cudaError_t::cudaSuccess;

}

}




