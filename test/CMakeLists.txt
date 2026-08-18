
# =============================================================================

# Define the targets.

# Bundle the source files into a variable.
set(TESTS_SRC
        pivots-selection.cpp
        partitioning.cpp
        disk-backend.cpp
        ranking.cpp
        rank-propagation.cpp
        rank-propagation.cu
        end-to-end.cpp
)

# Define the tester library.
add_library(tests STATIC ${TESTS_SRC})


# Set the include directories for the targets.
target_include_directories(tests PRIVATE ${INCLUDE_DIR} ${TESTS_INCLUDE} ${EXT_INCLUDE})
