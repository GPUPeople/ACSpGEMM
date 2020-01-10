# AC-SpGEMM

Public repository holding source code for AC-SpGEMM

## General Information

This repository holds the implementation for AC-SpGEMM and three different executables used for testing.
This framework can be used to setup AC-SpGEMM, run it with different matrices, test its performance against cuSparse (the other frameworks are not included in this public repository).

---


## Tested Setup
### Requirements
NVIDIA CUDA Toolkit, a C++14 enabled compiler, CMake, a reasonable modern GPU as well as NVIDIA CUB

Framework has been tested with the following versions:
- CUDA 9.1/9.2/10.0 
- gcc-6/gcc-7 or MVSC (VS17) (c++14 support required)
- CMake 3.2 or higher
- GPU with CC 6.1
- CUB v1.8.0

Set `CUDACXX` in `setup.sh` and `CUDA_INCLUDE_PATH` as well as `CUDA_BUILD_CCXX` in `CMakeLists.txt` to the appropriate values for your system.
To download the matrices used for evaluation, download the `ssgui` tool from [SuiteSparse](https://sparse.tamu.edu/interfaces), filtering the set of all matrices by setting the minimum number of non-zeros to `10000`(i.e. only testing matrices of non-trivial size). 

### Linux:
Run `setup.sh` from top level directory.

### Windows:
Download CUB (tested with v1.8.0), create folder `external` in `include` folder and extract contents into this folder.
Create folder `build` in top-level directory.
Setup CMake to setup project into this build folder.

---

## Running project
After building the project, 3 executables should be present
* HostTest
* performTestCase
* checkBitStability

### Reproducing Results
To get the timing and memory data from the framework, simply run `test/runall.sh(Linux)/.bat(Win)`, changing the `folder` parameter to a folder holding the matrices in `.mtx` format to test all the matrices in this folder with AC-SpGEMM and cuSparse.
If detailed timing results and memory results should be required, the last parameter of the `Multiply` call has to be changed to `true`, this will disable streams, print out more information and gather individual stage timings and memory measurements.

### HostTest
Sourcecode available in `source/main.cpp`, can be called as
`./HostTest <matrix.mtx> [deviceID] [testing]` (e.g. `./HostTest 1138_bus.mtx 0 0`).
Testing enabled will check the output matrix of AC-SpGEMM vs cuSparse.

### performTestCase
Sourcecode available in `source/performTestCase.cpp`, can be called as
`./performTestCase <folder_with_matrices> <runtests> <deviceID> <continue_run> <run_traits> <bitmask for run selection> <datatype (f/d)>`
* folder_with_matrices : A folder containing matrices to test
* runtests : Enabled (1) if tests should be run, (0) only statistics run
* deviceID : Which device to use (e.g. 0)
* continue_run : Set to true for testing
* run_traits : These traits are used to select a configuration, the parameters are 
  * Threads (256)
  * BlocksPerMP (3)
  * NNZPerThread (2)
  * InputElementsPerThreads (4)
  * RetainElementsPerThreads (4)
  * MaxChunksToMerge (16)
  * MaxChunksGeneralizedMerge (512)
  * MergePathOptions (8)
* bitmask for run selection : Select which approaches to run
  * Bit 0: cuSparse
  * Bit 1: AC-SpGEMM
  * E.g.: To run both cuSparse + acSpGEMM : 3
* datatype (f/d) : either float(f) or double(d)

Example: `./performTestCase folder 1 0 1 256,3,2,4,4,16,512,8 3 f`.
This testcase is typically called with a script (e.g. /test/runall.sh) which repeatedly calls this executable until all aproaches were tested on all matrices in the folder.
The reason behind this strategy is to get a fresh launch for each individual testcase, such that a failing testcase is not able to use up resources for another testcase.
To run a full testrun on a folder, change the `folder` in the `test/runall.sh/bat` script and run the script.

### checkBitStability
Sourcecode available in `source/checkBitStability.cpp`.
Works similarly to `performTestCase`, checks the approaches for bitstable results.

---
## Important Information
AC-SpGEMM is highly configurable as can be seen with the traits in the `performTestCase`, these traits are implemented as template parameters.
Hence, for all combinations used, the **respective instantiation must be present**.
Instantiations can be created by modifying the call to `Multiply` in `source/GPU/Multiply.cu` in line 781, which is given as
```cpp
bool called = 
	EnumOption<256, 256, 128, // Threads
	EnumOption<3, 4, 1, // BlocksPerMP
	EnumOption<2, 2, 1, // NNZPerThread
	EnumOption<4, 4, 1, // InputElementsPerThreads
	EnumOption<4, 4, 1, // RetainElementsPerThreads
	EnumOption<16, 16, 8, // MaxChunksToMerge
	EnumOption<256, 512, 256, // MaxChunksGeneralizedMerge
	EnumOption<8, 8, 8, // MergePathOptions
	EnumOption<0, 1, 1>>>>>>>>> // DebugMode
			::call(Selection<MultiplyCall<DataType>>(call), scheduling_traits.Threads, scheduling_traits.BlocksPerMp, scheduling_traits.NNZPerThread, scheduling_traits.InputElementsPerThreads, scheduling_traits.RetainElementsPerThreads, scheduling_traits.MaxChunksToMerge, scheduling_traits.MaxChunksGeneralizedMerge, scheduling_traits.MergePathOptions, (int)Debug_Mode);
```
This expanding template will instantiate variants of `MultiplyCall` with the parameters specified in `EnumOption<Start, End, Step>`, so each EnumOption describes all the possible values for a certain property and all different configurations will be instantiated (e.g. BlocksPerMP with `EnumOption<3, 4, 1,` will instantiate the template call with BlocksPerMP=3 and BlocksPerMP=4)

---
# FAQ
For any questions please directly contact Martin Winter <martin.winter@icg.tugraz.at>
