
# =============================================================================

# Benchmark harness.
# `cmake --build . --target <bench_baseline>`.

include(FetchContent)

# Core's feature macros must not reach a baseline binary. Clearing the property
# is the only way to un-inherit it.
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "")

if(NOT COMPILE_FLAGS)
    message(FATAL_ERROR "COMPILE_FLAGS is empty: benchmark/ was configured before the root set it.")
endif()

# Fetched sources land under `${CMAKE_BINARY_DIR}/_deps` (FetchContent's default).
# Do not pin this to `${CMAKE_SOURCE_DIR}/build/_deps`.


# ======== `libsais`: CPU baseline SACA ========
#
# If no network at configure time, use a local copy:
#   cmake -DFETCHCONTENT_SOURCE_DIR_LIBSAIS=/path/to/libsais ..

set(LIBSAIS_VERSION "2.10.4")

FetchContent_Declare(libsais
    URL https://github.com/IlyaGrebnov/libsais/archive/refs/tags/v${LIBSAIS_VERSION}.tar.gz
    URL_HASH SHA256=94aa88f9e29f8812214ecfa6b55f5dca14c7d8a427409c062f734a61ca4c6931
)

# Upstream's own options. `FORCE` because upstream declares them as cache
# variables with its own defaults.
set(LIBSAIS_USE_OPENMP ON CACHE BOOL "" FORCE)
set(LIBSAIS_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(libsais)


# ======== `libcubwt`: GPU baseline SACA ========
#
# Offline: -DFETCHCONTENT_SOURCE_DIR_LIBCUBWT=/path/to/libcubwt
set(LIBCUBWT_VERSION "1.0.0")   # v1.0.0 is the last release supporting SA construction.

FetchContent_Declare(libcubwt
    URL https://github.com/IlyaGrebnov/libcubwt/archive/refs/tags/v${LIBCUBWT_VERSION}.tar.gz
    URL_HASH SHA256=346b8f6a25dac67295b741d5f4382fad537b2ca3b2a5f32a2e5bc66d89473323
)

FetchContent_MakeAvailable(libcubwt)

# Upstream ships no CMakeLists.txt
add_library(libcubwt STATIC ${libcubwt_SOURCE_DIR}/libcubwt.cu)
set(LIBCUBWT_INC ${CMAKE_CURRENT_BINARY_DIR}/libcubwt_include)
file(COPY ${libcubwt_SOURCE_DIR}/libcubwt.cuh DESTINATION ${LIBCUBWT_INC}/libcubwt)
target_include_directories(libcubwt PUBLIC ${LIBCUBWT_INC})
set_target_properties(libcubwt PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES         ${CUDA_ARCH})


# Provenance (BENCH_COMMIT) is computed once at the root, so that our method's
# driver (in `src/`) carries the same stamp as the baselines here.


# ======== Shared driver settings ========
#
add_library(bench_settings INTERFACE)
target_compile_options(bench_settings INTERFACE ${COMPILE_FLAGS})
# The same list the root gives our own driver, not a second copy of it: these
# drifted apart once already, and a baseline stamping different provenance from
# ours is exactly the kind of difference nobody notices.
target_compile_definitions(bench_settings INTERFACE ${BENCH_PROVENANCE})


# ======== Drivers ========

add_executable(bench_libsais bench_libsais.cpp)
# OpenMP::OpenMP_CXX explicitly: libsais exports OpenMP::OpenMP_C, whose
# -fopenmp is guarded by $<COMPILE_LANGUAGE:C> and so never reaches this .cpp,
# which calls omp_get_dynamic().
target_link_libraries(bench_libsais PRIVATE bench_settings libsais OpenMP::OpenMP_CXX)

add_executable(bench_libcubwt bench_libcubwt.cpp)
target_link_libraries(bench_libcubwt PRIVATE bench_settings libcubwt CUDA::cudart)

install(TARGETS bench_libsais bench_libcubwt DESTINATION bin)
