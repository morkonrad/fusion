# Cuda based image fusion

What is this ? 
--------------
It's parallelized implementation of multi-scale image decomposition algorithms. Current impl. includes CUDA based Lift-DWT and Mallat-DWT. Additionally there is also an initial impl. of Mallat-DWT parallelized with OpenMP for multi-core CPUs. Finally, current impl. accelerates data transfers on Jetson TX2.

Features
----------
- Asynchronous cuda-streams.
- Pipelined-overlap I/O transfers with a computation kernels.

Requierments ?
---------------
1. C++11 compiler
2. CMake 3.x
3. CUDA lib and headers

How to build ?
---------------
  
How to use it ?
----------------


References
------------
1) Real-time fusion of visible and thermal infrared images in surveillance applications on SoC hardware. https://doi.org/10.1117/12.2325391

2) Implementation of the DWT in a GPU through a Register-based Strategy. https://doi.org/10.1109/TPDS.2014.2384047

3) https://github.com/PabloEnfedaque/CUDA_DWT_RegisterBased

