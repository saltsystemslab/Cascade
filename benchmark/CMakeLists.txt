
# =============================================================================

# Benchmark harness.
# `cmake --build . --target bench_baseline`.

include(FetchContent)

# Fetched sources land under `build/_deps/`.
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/build/_deps)


# ======== `libsais`: CPU baseline SACA ========
#
# If no network at configure time, use a local copy:
#   cmake -DFETCHCONTENT_SOURCE_DIR_LIBSAIS=/path/to/libsais ..

set(LIBSAIS_VERSION "2.10.4")

FetchContent_Declare(libsais
    URL https://github.com/IlyaGrebnov/libsais/archive/refs/tags/v${LIBSAIS_VERSION}.tar.gz
    # URL_HASH SHA256=<hash>    # TODO: pin before the artifact submission
)

# Upstream's own options. `FORCE` because upstream declares them as cache
# variables with its own defaults.
set(LIBSAIS_USE_OPENMP ON CACHE BOOL "" FORCE)
set(LIBSAIS_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(libsais)


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

target_link_libraries(bench_libsais PRIVATE bench_record libsais)
target_compile_options(bench_libsais PRIVATE ${COMPILE_FLAGS})

install(TARGETS bench_libsais DESTINATION bin)
