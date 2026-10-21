# =============================================================================

# Define the targets.

# Suffix-array converter for query indexes; host-only.
add_executable(sa_convert sa_convert.cpp)


# Set the include directories for the targets.

target_include_directories(sa_convert PRIVATE ${INCLUDE_DIR})
