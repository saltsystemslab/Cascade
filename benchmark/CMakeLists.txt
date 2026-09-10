
# =============================================================================

# Benchmark harness.
# `cmake --build . --target <bench_baseline>`.

include(FetchContent)

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


# ======== Record schema ========
#
# Header-only interface target.

add_library(bench_record INTERFACE)
target_include_directories(bench_record INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})


# ======== Provenance ========
#
# Tolerant of a missing .git.
execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE REPO_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)
if(NOT REPO_COMMIT)
    set(REPO_COMMIT "unknown")
endif()


# ======== Driver ========
#
add_executable(bench_libsais bench_libsais.cpp)

target_compile_definitions(bench_libsais PRIVATE
    BENCH_REPO_COMMIT="${REPO_COMMIT}"
    BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

target_link_libraries(bench_libsais PRIVATE bench_record libsais)
target_compile_options(bench_libsais PRIVATE ${COMPILE_FLAGS})

install(TARGETS bench_libsais DESTINATION bin)
