cmake_minimum_required(VERSION 3.25)
project(qwen-lora)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# for env witout setting utf-8 for Chinese 
if(WIN32)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>") #https://learn.microsoft.com/en-us/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8?view=msvc-170
    #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif()

if(WIN32) #default compiler on win is msvc. I've not verifed it with clang-cl yet.
    add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
    add_compile_options(/Zc:__cplusplus) # Add /Zc:__cplusplus flag for Visual Studio to properly set the __cplusplus macro
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Choose the configuration types" FORCE)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL") # refer to https://developercommunity.visualstudio.com/t/debug-build-works-but-release-build-failsas-well-a/428160
endif()

# Workaround for an MSVC compiler issue in some versions of Visual Studio 2022.
# The issue involves a null dereference to a mutex. For details, refer to link https://github.com/microsoft/STL/wiki/Changelog#vs-2022-1710
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# find_package(OpenVINO REQUIRED COMPONENTS Runtime) # should use OpenVINOGenAI
find_package(OpenVINOGenAI REQUIRED
    PATHS
        ${INTEL_OPENVINO_DIR} # get the environment variable
    NO_CMAKE_FIND_ROOT_PATH
)
if(WIN32)
    message("Original: ${CMAKE_CURRENT_SOURCE_DIR}")
    string(REPLACE "/" "\\\\" ESCAPED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    message("Escaped: ${ESCAPED_SOURCE_DIR}")
endif()

configure_file(
    config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

set(HEADER_FILES utils.h)
set(SOURCE_FILES main.cpp)

add_executable(qwen-lora ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(qwen-lora
    PRIVATE openvino::genai
)

target_include_directories( qwen-lora PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
