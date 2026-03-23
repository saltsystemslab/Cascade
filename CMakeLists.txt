
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
set(SUPPRESS_WARNING_FLAGS -Wno-unused-function)

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
set(OPTIMIZE_FLAGS -funroll-loops -march=native -mavx)

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


# Set path for modules required to search for existing packages in the system.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Turn on position-independent code, if static libraries are used in the linked
# dynamic libraries, then without PIC the build may fail.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Search and load the GPU-related libraries, tools, compilers etc.
find_package(CUDAToolkit REQUIRED)

# Search and load setting for the `zlib` library. The library is required to seamlessly adapt
# the `kseq` library to gzip-compressed files.
find_package(ZLIB REQUIRED)

# Module required to download and install external projects.
include(ExternalProject)

set(EXT_LIB ${CMAKE_SOURCE_DIR}/external/lib/)
set(EXT_INCLUDE ${CMAKE_SOURCE_DIR}/external/include/)

file(MAKE_DIRECTORY ${EXT_LIB} ${EXT_INCLUDE})


# The `Debug` configuration optimizes the program for debugging and enables full debug information.
# The `Release` configuration enables most compiler optimizations for speed and defines `NDEBUG`
# (No Debug) which will remove all traces of the standard library assert calls.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()


# Add subdirectory `src` to the build; CMake will open `src/CMakeLists.txt` for such.
add_subdirectory(src)
