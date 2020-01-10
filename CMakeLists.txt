cmake_minimum_required(VERSION 3.8)
project(acSpGEMM LANGUAGES CXX CUDA)

# Set CUDA Include path on Linux
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(include)
include_directories(include/external)
include_directories(externals)

# Choose for which CC to build and if to enable Debug
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" FALSE)
option(CUDA_BUILD_CC60 "Build with compute capability 6.0 support" FALSE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" TRUE)
option(CUDA_BUILD_CC70 "Build with compute capability 7.0 support" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

# CUDA Flags
if (CUDA_BUILD_CC50)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
endif ()
if (CUDA_BUILD_CC52)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
endif ()
if (CUDA_BUILD_CC60)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60")
endif ()
if (CUDA_BUILD_CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
endif ()
if (CUDA_BUILD_CC70)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
endif ()

string(APPEND CMAKE_CUDA_FLAGS " -std=c++14 -Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

# CXX Flags
if(WIN32)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS "/wd4464 /wd4514 /wd4820 /wd4668 /wd4574 /wd4571 /wd4324 /wd4710 /wd4711 /wd4365 /wd4515 /wd4201 /wd4267 /wd5027 /wd4626")
endif()
else()
set(GCC_COVERAGE_LINK_FLAGS  "-lstdc++fs")
string(APPEND CMAKE_CXX_FLAGS "-std=c++14 -mlzcnt -Wno-unknown-pragmas")
endif()


set(ACSPGEMMHEADERS
	include/common.cuh 
	include/common.h 
	include/Compare.h 
	include/consistent_memory.h 
	include/COO.h 
	include/CSR.h 
	include/CustomExceptions.h 
	include/dCSR.h 
	include/default_scheduling_traits.h 
	include/dVector.h 
	include/execution_stats.h 
	include/memory_space.h 
	include/MergeCaseOffsets.h 
	include/meta_utils.h 
	include/multi_arch_build.h 
	include/Multiply.h 
	include/Transpose.h 
	include/Vector.h
	include/device/acSpGEMM_ChunksToCSR.cuh 
	include/device/acSpGEMM_DetermineBlockStarts.cuh 
	include/device/acSpGEMM_MergeGeneralized.cuh 
	include/device/acSpGEMM_MergeMaxChunks.cuh 
	include/device/acSpGEMM_MergeSimple.cuh 
	include/device/acSpGEMM_SpGEMM.cuh 
	include/device/ARowStorage.cuh 
	include/device/Chunk.cuh 
	include/device/consistent_gpu_memory.h 
	include/device/HelperFunctions.cuh 
	include/device/MultiplyKernels.h
	include/device/SortAndCombine.cuh 
	include/device/WorkDistribution.cuh 
	include/devicetools/consistent_memory.h
	include/devicetools/error.h 
	include/devicetools/event.h
	include/devicetools/memory.h
	include/devicetools/memory_space.h
	include/devicetools/stream.h 
	include/devicetools/unique_handle.h
)


# AcSpGEMM Library
add_library(acSpGEMM
				source/device/Transpose.cu
				source/device/memory.cpp
				source/device/Compare.cu
				source/COO.cpp
				source/CSR.cpp
				source/dCSR.cpp
				source/device/Multiply.cu
				${ACSPGEMMHEADERS}
)

# Comparison implementations for cuSparse
set(COMP_SOURCES "externals/cusparse/source/cuSparseMultiply.cu")

add_library(comp_implementations
				${COMP_SOURCES}
)

# Executable to run a single test
add_executable(HostTest 
				source/main.cpp
)

# Executable to perform a complete testrun
add_executable(performTestCase
				source/performTestCase.cpp
)

# Executable to test bit stability
add_executable(checkBitStability
				source/checkBitStability.cpp
)

if(WIN32)
set_property(TARGET performTestCase PROPERTY CXX_STANDARD 17)
set_property(TARGET checkBitStability PROPERTY CXX_STANDARD 17)
endif()

# Link Libraries
target_link_libraries(HostTest acSpGEMM comp_implementations ${CUDA_cudart_static_LIBRARY} cusparse )
target_link_libraries(performTestCase acSpGEMM comp_implementations ${CUDA_cudart_static_LIBRARY} cusparse ${GCC_COVERAGE_LINK_FLAGS} )
target_link_libraries(checkBitStability acSpGEMM comp_implementations ${CUDA_cudart_static_LIBRARY} cusparse ${GCC_COVERAGE_LINK_FLAGS} )
