
# =============================================================================

# Specify the minimum version of CMake to use.
# CMake can have different behaviors (policies) based on its version used.
cmake_minimum_required(VERSION 3.17)


# Name the project, its version, and languages used in it.
set(PROJECT_NAME caps_sa_gpu)
project(${PROJECT_NAME}
        VERSION 0.0.1
        LANGUAGES CUDA CXX C
)


# Fix language standards, and set hard requirements for such.
# All targets defined from this point onward will pick up these requirements.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Turn off using platform-specific compiler standards.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

# Fix minimum compiler versions.
set(GCC_VERSION_MIN "9.1")
set(CLANG_VERSION_MIN "9.0")

if(CLANG)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    # set(CMAKE_LINKER_TYPE LLD)
endif()


# Bundle the warning flags that we want to pass on to the compiler.
# Reference: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
set(WARNING_FLAGS -Wall -Wextra) #-Wpedantic)
set(SUPPRESS_WARNING_FLAGS -Wno-unused-function -Wno-unknown-pragmas)

# Warning flags to be suppressed with clang.
set(CLANG_SUPPRESS_WARNING_FLAGS "")

if(CLANG)
    list(APPEND SUPPRESS_WARNING_FLAGS ${CLANG_SUPPRESS_WARNING_FLAGS})
endif()

# Bundle debugging information flags that we want to conditionally
# pass on to the compiler.
# Reference: https://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html.
set(DEBUG_FLAGS -ggdb)

# Bundle the extra optimization flags (not associated with the `-O` levels)
# that we want to pass on to the compiler.
# Reference: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
set(OPTIMIZE_FLAGS -mavx2)

# Bundle miscellaneous required flags.
set(MISC_FLAGS "")


# Accumulate the compile- and link-flags.
set(COMPILE_FLAGS "")
set(LINK_FLAGS "")

# Specify the warnings and the extra optimization flags to the compiler for the target library.
list(APPEND COMPILE_FLAGS ${WARNING_FLAGS} ${SUPPRESS_WARNING_FLAGS} ${OPTIMIZE_FLAGS} ${MISC_FLAGS})

if(DEBUG_INFO)
    list(APPEND COMPILE_FLAGS ${DEBUG_FLAGS})
endif()

# list(APPEND COMPILE_FLAGS --extended-lambda)
list(APPEND COMPILE_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
list(APPEND COMPILE_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)


# Add the required preprocessor definitions (`#define`s) to pass on.
add_compile_definitions(PROJECT_VERSION=${CMAKE_PROJECT_VERSION})
add_compile_definitions(CMAKE_EXPORT_COMPILE_COMMANDS)  # For better `clangd` scanning.
add_compile_definitions(_GNU_SOURCE)    # For `O_DIRECT` support.

if(ELL)
    add_compile_definitions(ELL=${ELL})
endif()

if(TESTS)
    add_compile_definitions(TESTS)
endif()

if(VALIDATE)
    add_compile_definitions(VALIDATE)
endif()

if(STATS)
    add_compile_definitions(STATS)
endif()

if(PROFILE)
    add_compile_definitions(PROFILE)
endif()


# Set path for modules required to search for existing packages in the system.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Turn on position-independent code, if static libraries are used in the linked
# dynamic libraries, then without PIC the build may fail.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add CUDA runtime flag.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -lcudart")


# Search and load the GPU-related libraries, tools, compilers etc.
find_package(CUDAToolkit REQUIRED)

# Search and load the `OpenMP` runtime libraries; required for testing purposes.
find_package(OpenMP REQUIRED)

# Search and load setting for the `zlib` library. The library is required to seamlessly adapt
# the `kseq` library to gzip-compressed files.
find_package(ZLIB REQUIRED)

# Module required to download and install external projects.
include(ExternalProject)

set(EXT_LIB     ${CMAKE_SOURCE_DIR}/external/lib/)
set(EXT_INCLUDE ${CMAKE_SOURCE_DIR}/external/include/)

file(MAKE_DIRECTORY ${EXT_LIB} ${EXT_INCLUDE})


# Prepare the `concurrentqueue` library. It provides fast multi-producer, multi-
# consumer lock-free concurrent queues.
set(CONCURRENTQUEUE_MIN_VERSION "1.0.5")
message("Build system will fetch and install moodycamel::ConcurrentQueue.")
ExternalProject_Add(prj_concurrentqueue
    DOWNLOAD_DIR        ${CMAKE_SOURCE_DIR}/external
    DOWNLOAD_COMMAND    wget -O concurrentqueue-${CONCURRENTQUEUE_MIN_VERSION}.tar.gz https://github.com/cameron314/concurrentqueue/archive/refs/tags/v${CONCURRENTQUEUE_MIN_VERSION}.tar.gz &&
                        tar -xzf concurrentqueue-${CONCURRENTQUEUE_MIN_VERSION}.tar.gz &&
                        rm concurrentqueue-${CONCURRENTQUEUE_MIN_VERSION}.tar.gz

    SOURCE_DIR          ${CMAKE_SOURCE_DIR}/external/concurrentqueue-${CONCURRENTQUEUE_MIN_VERSION}
    BUILD_IN_SOURCE     TRUE
    INSTALL_DIR         ${CMAKE_SOURCE_DIR}/external
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     sh -c "mkdir -p ${EXT_INCLUDE}/concurrentqueue && cp -rf concurrentqueue.h blockingconcurrentqueue.h lightweightsemaphore.h ${EXT_INCLUDE}/concurrentqueue/"
)


# The `Debug` configuration optimizes the program for debugging and enables full debug information.
# The `Release` configuration enables most compiler optimizations for speed and defines `NDEBUG`
# (No Debug) which will remove all traces of the standard library assert calls.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()


# Add required subdirectories to the build.
add_subdirectory(src)
add_subdirectory(test)


# Set some CUDA-specific properties.
set_property(TARGET core            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET tests           PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET libcubwt        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET libcubwt_driver PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test            PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_ARCH "120-real")
set_property(TARGET core            PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET tests           PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET libcubwt        PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET libcubwt_driver PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET test            PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})


# Define dependencies between the targets.
add_dependencies(${PROJECT_NAME} core)
add_dependencies(core prj_concurrentqueue)


# Set the compile flags for the targets.

target_compile_options(core PRIVATE ${COMPILE_FLAGS})
target_compile_options(libcubwt PRIVATE ${COMPILE_FLAGS})

target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})
target_compile_options(libcubwt_driver PRIVATE ${COMPILE_FLAGS})
target_compile_options(test PRIVATE ${COMPILE_FLAGS})

# Pass OpenMP flag through the CUDA compiler down to the host compiler.
# `nvcc` does not recognize OpenMP directives.
target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)


# Link appropriate libraries and targets.

# Link to `zlib`, required to support `gzip` files.
target_link_libraries(core PRIVATE ZLIB::ZLIB)
target_link_libraries(libcubwt_driver PRIVATE ZLIB::ZLIB)

# Link to CUDA runtime.
target_link_libraries(core PRIVATE CUDA::cudart)

# Link to OpenMP for host-side parallel tasks.
target_link_libraries(core PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)

# Link to the tester.
target_link_libraries(core PRIVATE tests)

# Link the executables to the required libraries.
target_link_libraries(${PROJECT_NAME} PRIVATE core)
target_link_libraries(libcubwt_driver PRIVATE libcubwt)

# Link the tester.
target_link_libraries(test PRIVATE core)


# Set the link flags for the targets.

target_link_libraries(core PRIVATE ${LINK_FLAGS})
target_link_libraries(libcubwt PRIVATE ${LINK_FLAGS})

target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_FLAGS})
target_link_libraries(libcubwt_driver PRIVATE ${LINK_FLAGS})
target_link_libraries(test PRIVATE ${LINK_FLAGS})


# Set the installation directory. The parent directory is `/usr/local/` by default, and the
# user may set it to a different path by passing a value for `CMAKE_INSTALL_PREFIX`. The target
# executable is installed inside a sub-directory named `bin` within the parent directory.
install(TARGETS ${PROJECT_NAME} libcubwt_driver
        DESTINATION bin
        )

install(TARGETS core
  DESTINATION lib
)
