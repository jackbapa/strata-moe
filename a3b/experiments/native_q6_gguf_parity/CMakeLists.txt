cmake_minimum_required(VERSION 3.24)
project(strata_native_q6_gguf_parity LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

get_filename_component(STRATA_MOE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(LLAMA_SOURCE "${STRATA_MOE_ROOT}/third_party/llama.cpp" CACHE PATH "Pinned llama.cpp source")
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_SOURCE}" llama)
find_package(CUDAToolkit REQUIRED)

# The pinned GGML CUDA sources contain A3B hooks. Build their actual link
# dependencies exactly as a3b/CMakeLists.txt does, without modifying the
# selected service target or the original G:/Strata checkout.
add_library(strata_a3b_native_probe_deps STATIC
  "${STRATA_MOE_ROOT}/a3b/strata_native_experts.cu"
  "${STRATA_MOE_ROOT}/a3b/strata_prefill_mmq_adapter.cu"
  "${STRATA_MOE_ROOT}/src/prefill/moe_mmq.cu"
  "${STRATA_MOE_ROOT}/src/kernels/cuda/iq_kernels.cu"
  "${STRATA_MOE_ROOT}/src/kernels/cuda/fused_gdn.cu")
set_source_files_properties("${STRATA_MOE_ROOT}/src/kernels/cuda/fused_gdn.cu"
  PROPERTIES COMPILE_OPTIONS "-use_fast_math")
set_source_files_properties(
  "${STRATA_MOE_ROOT}/a3b/strata_prefill_mmq_adapter.cu"
  "${STRATA_MOE_ROOT}/src/prefill/moe_mmq.cu"
  PROPERTIES COMPILE_OPTIONS "-use_fast_math;--extended-lambda")
set_target_properties(strata_a3b_native_probe_deps PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_include_directories(strata_a3b_native_probe_deps PRIVATE
  "${STRATA_MOE_ROOT}/include" "${STRATA_MOE_ROOT}/src/kernels/cuda"
  "${STRATA_MOE_ROOT}/third_party/ggml"
  "${LLAMA_SOURCE}/ggml/src" "${LLAMA_SOURCE}/ggml/src/ggml-cuda")
if(GGML_CUDA_GRAPHS)
  target_compile_definitions(strata_a3b_native_probe_deps PRIVATE GGML_CUDA_USE_GRAPHS)
endif()
target_link_libraries(strata_a3b_native_probe_deps PUBLIC ggml-base CUDA::cudart)
target_sources(ggml-cuda PRIVATE
  "${LLAMA_SOURCE}/ggml/src/ggml-cuda/strata-native-iq.cu")
target_include_directories(ggml-cuda PRIVATE
  "${STRATA_MOE_ROOT}/a3b" "${STRATA_MOE_ROOT}/include")
target_link_libraries(ggml-cuda PRIVATE strata_a3b_native_probe_deps)

set(NATIVE_MMVQ_SOURCE "${STRATA_MOE_ROOT}/src/kernels/cuda/native_mmvq.cu")
# Match original Strata's CUDA build of native_mmvq.cu.
set_source_files_properties("${NATIVE_MMVQ_SOURCE}" PROPERTIES COMPILE_OPTIONS "--use_fast_math")
add_executable(native_q6_gguf_parity
  native_q6_gguf_parity.cu
  "${NATIVE_MMVQ_SOURCE}")
set_target_properties(native_q6_gguf_parity PROPERTIES
  CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_include_directories(native_q6_gguf_parity PRIVATE
  "${STRATA_MOE_ROOT}/include"
  "${STRATA_MOE_ROOT}/src/kernels/cuda"
  "${LLAMA_SOURCE}/ggml/include"
  "${LLAMA_SOURCE}/ggml/src"
  "${LLAMA_SOURCE}/ggml/src/ggml-cuda")
if(GGML_CUDA_GRAPHS)
  # ggml_backend_cuda_context's conditional graph field must match ggml-cuda.
  target_compile_definitions(native_q6_gguf_parity PRIVATE GGML_CUDA_USE_GRAPHS)
endif()
target_link_libraries(native_q6_gguf_parity PRIVATE ggml ggml-cuda strata_a3b_native_probe_deps CUDA::cudart)
