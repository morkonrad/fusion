#pragma once 

#include "common.h"
#include <cstdint>
#include "cuda_runtime_api.h"

namespace FUSE_RULE
{

cudaError_t calc_fuse_maxabs(const DATATYPE* dev_in_vis,
                             const DATATYPE* dev_in_ir,
                             DATATYPE* dev_out_fused,
                             const std::uint32_t& rows,
                             const std::uint32_t& cols,
                             cudaStream_t& stream);

cudaError_t calc_fuse_avg(const DATATYPE* dev_in_vis,
                          const DATATYPE* dev_in_ir,
                          DATATYPE* dev_out_fused,
                          const std::uint32_t& rows,
                          const std::uint32_t& cols,
                          cudaStream_t& stream);

cudaError_t calc_fuse_sum(const DATATYPE* dev_in_vis,
                          const DATATYPE* dev_in_ir,
                          DATATYPE* dev_out_fused,
                          const std::uint32_t& rows,
                          const std::uint32_t& cols,
                          cudaStream_t& stream);

cudaError_t calc_fuse_sum_abs(const DATATYPE* dev_in_vis,
                              const DATATYPE* dev_in_ir,
                              DATATYPE* dev_out_fused,
                              const std::uint32_t& rows,
                              const std::uint32_t& cols,
                              cudaStream_t& stream);

cudaError_t calc_fuse_max(const DATATYPE* dev_in_vis,
                          const DATATYPE* dev_in_ir,
                          DATATYPE* dev_out_fused,
                          const std::uint32_t& rows,
                          const std::uint32_t& cols,
                          cudaStream_t& stream);
}

