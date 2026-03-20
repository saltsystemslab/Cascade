
# =============================================================================

# This controls the `-I` (include paths) switch of the compiler invocation.
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)


# Define the targets.

# Bundle the core source files into a variable.
set(PROJECT_SRC
        Parser.cpp
        SA.cu
)

# Define the core library.
# `STATIC` will force the creation of a static library (and so no runtime linking).
add_library(core STATIC ${PROJECT_SRC})

# Define an executable from the driver program.
add_executable(${PROJECT_NAME} main.cu)
add_executable(test test.cu)


# Set some CUDA-specific properties.
set_property(TARGET core            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test            PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_ARCH "70-real")
set_property(TARGET core            PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET test            PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})


# Define dependencies between the targets.


# Set the include directories for the targets.

target_include_directories(core PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
target_include_directories(test PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE})


# Set the compile flags for the targets.

target_compile_options(core PRIVATE ${COMPILE_FLAGS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})
target_compile_options(test PRIVATE ${COMPILE_FLAGS})


# Link appropriate libraries and targets.

# Link to `zlib`, required to support `gzip` files.
target_link_libraries(core PRIVATE ZLIB::ZLIB)

# Link the executable to the core library.
target_link_libraries(${PROJECT_NAME} PRIVATE core)

# Link the tester.
target_link_libraries(test PRIVATE core)


# Set the link flags for the targets.

target_link_libraries(core PRIVATE ${LINK_FLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_FLAGS})
target_link_libraries(test PRIVATE ${LINK_FLAGS})


# Set the installation directory. The parent directory is `/usr/local/` by default, and the
# user may set it to a different path by passing a value for `CMAKE_INSTALL_PREFIX`. The target
# executable is installed inside a sub-directory named `bin` within the parent directory.
install(TARGETS ${PROJECT_NAME}
        DESTINATION bin
        )

install(TARGETS core
  DESTINATION lib
)
