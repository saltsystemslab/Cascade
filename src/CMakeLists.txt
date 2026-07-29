
# =============================================================================

# This controls the `-I` (include paths) switch of the compiler invocation.
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(TESTS_INCLUDE ${CMAKE_SOURCE_DIR}/test)


# Define the targets.

# Bundle the core source files into a variable.
set(PROJECT_SRC
        GPU_SACA.cu
        Partition.cu
        Suffix_Buffer.cpp
        Staging_Slab.cu
        RAM_Backend.cpp
        Direct_File.cpp
        Spill_Store.cpp
)

# Define the core library and the `libcubwt` library for benchmarking.
add_library(core STATIC ${PROJECT_SRC})
add_library(libcubwt STATIC ${INCLUDE_DIR}/libcubwt/libcubwt.cu)        # Separate it out to a separate benchmarking cmake file.


# Define an executable from the driver programs.
add_executable(${PROJECT_NAME} main.cu)
add_executable(libcubwt_driver libcubwt_driver.cu)
add_executable(test test.cu)


# Set the include directories for the targets.

target_include_directories(core PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE} ${TESTS_INCLUDE})

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE} ${TESTS_INCLUDE})
target_include_directories(libcubwt_driver PRIVATE ${INCLUDE_DIR})
target_include_directories(test PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE})
