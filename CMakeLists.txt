# =====================================================================================================================
# 
# =====================================================================================================================
cmake_minimum_required(VERSION 3.16)
project(rocksdb-robust VERSION 1.0
                       DESCRIPTION "RocksDB wrapper that implements components of a robust LSM tree"
                       LANGUAGES CXX
)

include(ExternalProject)
message(STATUS "CXX : ${CMAKE_CXX_COMPILER}")

# =====================================================================================================================
# HEADER Options
# =====================================================================================================================
option(DEBUG "Debug mode on or off" OFF)
if(${DEBUG})
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# =====================================================================================================================
# HEADER Submodules
# =====================================================================================================================
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMOD_RESULT
    )
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()

# =====================================================================================================================
# HEADER RocksDB 
# =====================================================================================================================

ExternalProject_Add(rocksdb_static_lib
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/rocksdb"
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    USES_TERMINAL_CONFIGURE 1
    BUILD_COMMAND $(MAKE) -C ${PROJECT_SOURCE_DIR}/extern/rocksdb DISABLE_WARNING_AS_ERROR=true static_lib
)

add_library(rocksdb STATIC IMPORTED)
add_dependencies(rocksdb rocksdb_static_lib)
set_target_properties(rocksdb PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/extern/rocksdb/librocksdb.a)
target_include_directories(rocksdb INTERFACE "${PROJECT_SOURCE_DIR}/extern/rocksdb/include")

# =====================================================================================================================
# HEADER Dependencies
# =====================================================================================================================
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(LZ4 REQUIRED)
find_package(Threads REQUIRED)
find_package(zstd REQUIRED)
find_package(snappy REQUIRED)
target_link_libraries(rocksdb INTERFACE
    BZip2::BZip2
    LZ4::LZ4
    Threads::Threads
    ZLIB::ZLIB
    zstd::zstd
    snappy::snappy
    ${CMAKE_DL_LIBS}
)

add_subdirectory(${PROJECT_SOURCE_DIR}/extern/spdlog)

# =====================================================================================================================
# HEADER TmpDB
# =====================================================================================================================
file(GLOB_RECURSE
        TMPDB_SRCS
        CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/tmpdb/*.cpp
        ${CMAKE_SOURCE_DIR}/src/tmpdb/*.hpp
)

add_library(tmpdb OBJECT "${TMPDB_SRCS}")

set_target_properties(tmpdb PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
)

target_compile_features(tmpdb PUBLIC
    cxx_std_11
)

target_include_directories(tmpdb PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extern
    ${CMAKE_SOURCE_DIR}/extern/rocksdb/include/rocksdb
)

target_compile_options(tmpdb PUBLIC
    "-Wall"
    "-Wextra"
    "-fstack-protector-strong"
    "-fexceptions"
    "-fasynchronous-unwind-tables"
    $<$<BOOL:${DEBUG}>:-g2>
    $<IF:$<BOOL:${DEBUG}>,-O0,-O2>
)

if(${DEBUG})
    message(STATUS " Configuring with debug flags")
endif()

target_link_libraries(tmpdb PUBLIC
    rocksdb
    spdlog
)

# =====================================================================================================================
# Tools
# =====================================================================================================================

file(GLOB_RECURSE
    TOOLS_SRCS
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/tools/infrastructure/*.cpp
    ${CMAKE_SOURCE_DIR}/tools/infrastructure/*.hpp
)
add_library(tools OBJECT ${TOOLS_SRCS})

target_link_libraries(tools PUBLIC tmpdb)

# Tools executables
add_executable(db_builder ${CMAKE_SOURCE_DIR}/tools/db_builder.cpp)
target_link_libraries(db_builder tmpdb tools)

add_executable(db_runner ${CMAKE_SOURCE_DIR}/tools/db_runner.cpp)
target_link_libraries(db_runner tmpdb tools)

add_executable(db_cont_runner ${CMAKE_SOURCE_DIR}/tools/db_cont_runner.cpp)
target_link_libraries(db_cont_runner tmpdb tools)

add_executable(key_gen ${CMAKE_SOURCE_DIR}/tools/key_gen.cpp)
target_link_libraries(key_gen tmpdb tools)

# add_executable(compact_files_example ${CMAKE_SOURCE_DIR}/example/compact_files_example.cc)
# target_link_libraries(compact_files_example tmpdb)
