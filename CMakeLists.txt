cmake_minimum_required(VERSION 3.19)
project(maac_pipeline LANGUAGES CXX)
find_package(Threads REQUIRED)
set(ROCM_PATH /opt/rocm CACHE PATH "ROCm installation")
# Pass -DPython3_EXECUTABLE=... : the Python that has ROCm PyTorch and PyG.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Host code without GPU dependencies: graph files and golden model, FPGA
# control, NVMe -> FPGA loading, FPGA -> GPU transfers.
add_library(e2e_common
  src/workload.cpp src/fpga_host.cpp src/nvme_loader.cpp
  src/p2p_batch.cpp src/p2p_registration.cpp)
set_target_properties(e2e_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(e2e_common PUBLIC cxx_std_17)
# qdma_persistent_uapi.h, the ioctl header of the modified QDMA driver, comes from
# the patched driver source tree: <dma_ip_drivers>/QDMA/linux-kernel/driver/include
set(QDMA_INCLUDE_DIR "" CACHE PATH "QDMA/linux-kernel/driver/include of the patched dma_ip_drivers checkout")
if(NOT EXISTS "${QDMA_INCLUDE_DIR}/qdma_persistent_uapi.h")
  message(FATAL_ERROR "Pass -DQDMA_INCLUDE_DIR=<dma_ip_drivers>/QDMA/linux-kernel/driver/include (the patched driver)")
endif()
target_include_directories(e2e_common PUBLIC src "${QDMA_INCLUDE_DIR}")
target_link_libraries(e2e_common PUBLIC Threads::Threads)
target_compile_options(e2e_common PRIVATE -Wall -Wextra -Wpedantic)

# Python extension `_fpga_pipeline`. PyTorch ships the pybind11 and DLPack headers.
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import pathlib,torch; print(pathlib.Path(torch.__file__).parent / 'include')"
  OUTPUT_VARIABLE TORCH_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE TORCH_RESULT)
if(NOT TORCH_RESULT EQUAL 0 OR NOT EXISTS "${TORCH_INCLUDE}/pybind11/pybind11.h")
  message(FATAL_ERROR "Select the ROCm PyTorch Python with -DPython3_EXECUTABLE=...")
endif()
find_path(HIP_INCLUDE hip/hip_runtime_api.h HINTS ${ROCM_PATH}/include REQUIRED)
find_library(HIP_LIBRARY amdhip64 HINTS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 REQUIRED)
find_library(HSA_LIBRARY hsa-runtime64 HINTS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 REQUIRED)
Python3_add_library(_fpga_pipeline MODULE WITH_SOABI src/pipeline.cpp src/bindings.cpp)
target_link_libraries(_fpga_pipeline PRIVATE e2e_common ${HIP_LIBRARY} ${HSA_LIBRARY} ${CMAKE_DL_LIBS})
target_include_directories(_fpga_pipeline SYSTEM PRIVATE ${HIP_INCLUDE} ${TORCH_INCLUDE})
target_compile_definitions(_fpga_pipeline PRIVATE __HIP_PLATFORM_AMD__)
target_compile_options(_fpga_pipeline PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(_fpga_pipeline PROPERTIES CXX_VISIBILITY_PRESET hidden)
