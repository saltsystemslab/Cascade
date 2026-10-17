
# =============================================================================

# Define the targets.

# Bundle the core source files into a variable.
set(PROJECT_SRC
        GPU_SACA.cu
        Partition_ID_Map.cpp
        Suffix_Buffer.cpp
        Staging_Slab.cpp
        Direct_File.cpp
        Output_File.cpp
        Spill_Store.cpp
        Disk_Backend.cpp
        RAM_Backend.cpp
        Message_Pool.cpp
        Partition_Backend.cpp
        Residency_Plan.cpp
        Pipeline_Probe.cpp
        utility.cpp
)

# Define the core library.
add_library(core STATIC ${PROJECT_SRC})


# Define an executable from the driver programs.
add_executable(${PROJECT_NAME} main.cu)
add_executable(test test.cu)

# Benchmark driver for our own method. Placement is with `core` rather than in
# the independent `benchmark/` so that our definitions do not leak to the
# baselines, and building this does not drag in the baselines' network fetches.
add_executable(bench_gpu_saca bench_gpu_saca.cu)

# Offline structural statistics of a text, from its suffix array and a STATS
# run's log. Host-only, independent of `core`.
add_executable(sa_structure sa_structure.cpp)

# Query engine.
add_library(query_core STATIC Suffix_Index.cu Sample_Table.cu Query_Engine.cu Host_Search.cpp)
add_executable(query query.cu)
add_executable(test_query test_query.cu)


# Set the include directories for the targets.

target_include_directories(core PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE} ${TESTS_INCLUDE})

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE} ${TESTS_INCLUDE})
target_include_directories(test PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE})
target_include_directories(bench_gpu_saca PRIVATE ${INCLUDE_DIR} ${EXT_INCLUDE} ${TESTS_INCLUDE})
target_include_directories(sa_structure PRIVATE ${INCLUDE_DIR})
target_include_directories(query_core PRIVATE ${INCLUDE_DIR})
target_include_directories(query PRIVATE ${INCLUDE_DIR})
target_include_directories(test_query PRIVATE ${INCLUDE_DIR} ${TESTS_INCLUDE})
