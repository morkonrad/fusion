# CoopCL
[![Build status](https://ci.appveyor.com/api/projects/status/8cg021yesld25ykm?svg=true)](https://ci.appveyor.com/project/morkonrad/coopcl)

What is this ? 
--------------
It's header only library that supports collaborative CPU-GPU workload processing. It enables parallel and asynchronous tasks execution described by the task graph.

**Features:**
1. Task graph API+Runtime
2. Parallel+asynchronous tasks/kernels execution on CPU+GPU
3. Variable workload splitting, partial offload to GPU
4. Support for APUs and CPUs+dGPUs

Requierments ?
---------------
1. C++14 compiler
2. CMake 3.x
3. OpenCL 2.x headers and lib, support for CPU and GPU
3. GPU driver with OpenCL and SVM_FINE_GRAIN_BUFFER support
4. For unit-tests CTest

How to build ?
---------------
  1. git clone CoopCL /dst
  2. cd /dst
  3. mkdir build
  4. cd build
  5. cmake -G"Visual Studio 14 2015 Win64" .. 
  6. cmake --build . --config Release
  
For Windows, Visual Studio 2015 is a minimal tested version. For Linux it's tested with GCC 7.0 and Clang 5.0. In general, compiler must support C++14. 

After succesfull build you can call unit tests to check if they pass:  
 1. cd /clDriver
 2. ctest 
  
How to use it ?
----------------
After successful build and tests, the CoopCL should be ready to go. 

It's header only library so you need to only link whith your app.

Check sample usage/application below.

Example:
----------------
The following code executes simple task graph. Tasks B,C are executed asynchronously and in parallel on CPU and GPU:
```cpp
#include "clDriver.h"
#include <cassert>
#include <iostream>
#include <stdlib.h>

int main()
{
  //Simple task_graph consist of 4 tasks	
    /*
    <BEGIN>
     [A]
    /   \
  [B]   [C]
    \   /
     [D]
    <END>
    */
    //A = 10 
    //B(A) = 11 >> B=A+1
    //C(A) = 12 >> C=A+2
    //D(B,C) = 23 >> D=B+C	

	constexpr auto tasks = R"(
  kernel void kA(global int* A)                        
  {
  const int tid = get_global_id(0);                                                       
  A[tid] = 10;
  }

  kernel void kB(const global int* A,global int* B)                        
  {
  const int tid = get_global_id(0);                                                       
  B[tid] = A[tid]+1;
  }

  kernel void kC(const global int* A,global int* C)                        
  {
  const int tid = get_global_id(0);                                                       
  C[tid] = A[tid]+2;
  }

  kernel void kD(const global int* B,
  const global int* C,global int* D)                        
  {
  const int tid = get_global_id(0); 
  D[tid] = B[tid]+C[tid];
  }
  )";
  
coopcl::virtual_device device;	
  
const size_t items = 1024;  
auto mA = device.alloc<int>(items);
auto mB = device.alloc<int>(items);
auto mC = device.alloc<int>(items);
auto mD = device.alloc<int>(items);

coopcl::clTask taskA;
device.build_task(taskA,tasks, "kA");
	
coopcl::clTask taskB;
device.build_task(taskB, tasks, "kB");
taskB.add_dependence(&taskA);

coopcl::clTask taskC;
device.build_task(taskC,tasks, "kC");
taskC.add_dependence(&taskA);

coopcl::clTask taskD;
device.build_task(taskD, tasks, "kD");
taskD.add_dependence(&taskB);
taskD.add_dependence(&taskC);

const std::array<size_t, 3> ndr = { items,1,1 };
const std::array<size_t, 3> wgs = { 16,1,1 };
	
for (int i = 0;i < 10;i++) 
{		
	device.execute_async(taskA, 0.0f, ndr, wgs, mA); //100% CPU
	device.execute_async(taskB, 0.8f, ndr, wgs, mA, mB); //80% GPU, 20 % CPU
	device.execute_async(taskC, 0.5f, ndr, wgs, mA, mC); //50% GPU, 50 % CPU
	device.execute_async(taskD, 1.0f, ndr, wgs, mB, mC, mD); //100% GPU
	taskD.wait();
}
	
for (int i = 0;i < items;i++)
{
	const auto val = mD->at<int>(i);
	if (val != 23)
	{
		std::cerr << "Some error at pos i = " << i << std::endl;
		return -1;
	}
}

std::cout << "Passed,ok!" << std::endl;
return 0;
}
```

Current state
----------------
CoopCL is still in an early stage of development. It can successfully execute many tasks with a variable offload ratio on Intel and AMD platforms, but not yet with NVIDIA GPUs. Current NVIDIA drivers support only OpenCL 1.x. 

The extension for NVIDIA Platforms and multi-GPU is in progress.

**Tested systems:**

| HW-Vendor | CPU       | GPU     | GPU-Driver     | OS    | Platform          |
| --------- | --------- | ------- | -------------- | ----- | ----------------- |
| Intel+AMD | I7-3930k  | R9-290  | 2906.10        | win64 | Desktop dCPU+dGPU |
| Intel	    | I7-660U   | HD-520  | 26.20.100.7158 | win64 | Notebook APU      |
| Intel	    | I7-8700   | UHD-630 | 26.20.100.7158 | win64 | Notebook APU      |
| AMD	    | R5-2400GE | Vega-11 | 2639.5         | win64 | Notebook APU      |
| AMD	    | R7-2700U  | Vega-10 | 2639.5         | win64 | Notebook APU      |

References
------------
Please cite: CoopCL: Cooperative Execution of OpenCL Programs on Heterogeneous CPU-GPU Platforms. 

28th Euromicro International Conference on Parallel, Distributed and Network-based Processing
PDP2020 (accepted for publication)

