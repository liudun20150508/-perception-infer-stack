cmake_minimum_required(VERSION 3.20)
project(edge-infer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# ---- 定位 onnxruntime（Homebrew / 自编译 / vendored 三种来源，跨平台）----
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS
    /opt/homebrew/include/onnxruntime
    /opt/homebrew/opt/onnxruntime/include/onnxruntime
    /usr/local/include/onnxruntime
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include/onnxruntime
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64/include
    C:/onnxruntime/include
)
find_library(ORT_LIBRARY NAMES onnxruntime
  HINTS
    /opt/homebrew/lib
    /opt/homebrew/opt/onnxruntime/lib
    /usr/local/lib
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64/lib
    C:/onnxruntime/lib
)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "onnxruntime not found, install with: brew install onnxruntime")
endif()
message(STATUS "onnxruntime include: ${ORT_INCLUDE_DIR}")
message(STATUS "onnxruntime lib:     ${ORT_LIBRARY}")

add_library(edgeinfer STATIC
  src/logger.cpp
  src/thread_pool.cpp
  src/model_engine.cpp
  src/model_manager.cpp
  src/image_utils.cpp
  src/inference_server.cpp
)
target_include_directories(edgeinfer PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party
  ${ORT_INCLUDE_DIR}
)
target_link_libraries(edgeinfer PUBLIC ${ORT_LIBRARY} Threads::Threads)

add_executable(edge_demo src/main.cpp)
target_link_libraries(edge_demo PRIVATE edgeinfer)

# Windows：把 onnxruntime.dll 自动拷贝到可执行文件目录（开箱即跑）
if(WIN32)
  get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
  add_custom_command(TARGET edge_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_LIB_DIR}/onnxruntime.dll"
      $<TARGET_FILE_DIR:edge_demo>)
endif()

# MSVC：源码含 UTF-8 中文注释，必须指定源码编码，否则乱码/编译错误
if(MSVC)
  target_compile_options(edgeinfer PRIVATE /utf-8)
  target_compile_options(edge_demo PRIVATE /utf-8)
endif()

option(EDGE_ASAN "Build with AddressSanitizer (memory check)" OFF)
if(EDGE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
  message(STATUS "AddressSanitizer enabled")
endif()
