cmake_minimum_required(VERSION 3.16)
project(omnistream_engine VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

include(FetchContent)

# For testing phase
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG        v25.12.19
)
set(FLATBUFFERS_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB   OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH  OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)

# ZeroMQ & libuv (vcpkg native config)
find_package(ZeroMQ CONFIG REQUIRED)

find_package(libuv CONFIG REQUIRED)
find_path(UV_INCLUDE_DIR NAMES uv.h)

if(NOT UV_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find uv.h! Ensure libuv is installed via vcpkg.")
endif()

FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG        v4.10.0
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cppzmq)

# uWebSockets & uSockets 
FetchContent_Declare(
  uwebsockets
  GIT_REPOSITORY https://github.com/uNetworking/uWebSockets.git
  GIT_TAG        v20.62.0
)
FetchContent_MakeAvailable(uwebsockets)

FetchContent_Declare(
  usockets
  GIT_REPOSITORY https://github.com/uNetworking/uSockets.git
  GIT_TAG        v0.8.8
)
FetchContent_MakeAvailable(usockets)

# Build uSockets as a static lib (Windows-safe fixes)
add_library(usockets STATIC
  ${usockets_SOURCE_DIR}/src/eventing/epoll_kqueue.c
  ${usockets_SOURCE_DIR}/src/eventing/libuv.c
  ${usockets_SOURCE_DIR}/src/io_uring/io_context.c
  ${usockets_SOURCE_DIR}/src/crypto/openssl.c
  ${usockets_SOURCE_DIR}/src/bsd.c
  ${usockets_SOURCE_DIR}/src/context.c
  ${usockets_SOURCE_DIR}/src/loop.c
  ${usockets_SOURCE_DIR}/src/socket.c
  ${usockets_SOURCE_DIR}/src/udp.c
)

# Explicitly bind the uv.h include directory here
target_include_directories(usockets PUBLIC 
  ${usockets_SOURCE_DIR}/src
  ${UV_INCLUDE_DIR}
)

target_compile_definitions(usockets PUBLIC LIBUS_NO_SSL)

# Platform-specific event loop flags
if(APPLE)
  target_compile_definitions(usockets PUBLIC LIBUS_USE_KQUEUE)
elseif(WIN32)
  target_compile_definitions(usockets PUBLIC LIBUS_USE_LIBUV)
else()
  target_compile_definitions(usockets PUBLIC LIBUS_USE_EPOLL)
endif()

if(TARGET libuv::uv_a)
  target_link_libraries(usockets PUBLIC libuv::uv_a)
elseif(TARGET libuv::uv)
  target_link_libraries(usockets PUBLIC libuv::uv)
else()
  target_link_libraries(usockets PUBLIC uv)
endif()

# cpp-httplib
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.15.3
)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(httplib)

# Engine executable
add_executable(engine
  src/main.cpp
  src/zmq_receiver.cpp
  src/ws_broadcaster.cpp
  src/mmap_log.cpp
  src/pipeline.cpp
  src/metrics_server.cpp
  src/anomaly_detector.cpp  
  src/replay_engine.cpp      ]
)
target_include_directories(engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/incl
  #${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
  ${flatbuffers_SOURCE_DIR}/include          
  ${cppzmq_SOURCE_DIR}
  ${uwebsockets_SOURCE_DIR}/src
  ${usockets_SOURCE_DIR}/src
  ${httplib_SOURCE_DIR}                      
)

target_link_libraries(engine PRIVATE
  usockets
  libzmq 
)

# Cross-platform compiler flags for Engine
if(MSVC)
  target_compile_options(engine PRIVATE /O2 /W4)
else()
  target_compile_options(engine PRIVATE -O2 -Wall -Wextra)
endif()

# Tests
enable_testing()

add_executable(ring_buffer_test tests/ring_buffer_test.cpp)
target_include_directories(ring_buffer_test PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/incl 
  #${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(adaptive_sampler_test tests/adaptive_sampler_test.cpp)
target_include_directories(adaptive_sampler_test PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/incl 
  #${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(anomaly_detector_test tests/anomaly_detector_test.cpp src/anomaly_detector.cpp)
target_include_directories(anomaly_detector_test PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/incl 
  #${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${CMAKE_CURRENT_SOURCE_DIR}/generated 
  ${flatbuffers_SOURCE_DIR}/include
)
# Apply Windows-Safe Testing Flags
if(MSVC)
  target_compile_options(ring_buffer_test PRIVATE /O2 /W4)
  target_compile_options(adaptive_sampler_test PRIVATE /O2 /W4)
  target_compile_options(anomaly_detector_test PRIVATE /O2 /W4)
else()
  # Use Thread Sanitizer only on Linux/Mac
  target_compile_options(ring_buffer_test PRIVATE -O2 -Wall -Wextra -fsanitize=thread)
  target_link_options(ring_buffer_test PRIVATE -fsanitize=thread)
  target_compile_options(adaptive_sampler_test PRIVATE -O2 -Wall -Wextra)
  target_compile_options(anomaly_detector_test PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME ring_buffer_stress COMMAND ring_buffer_test 5)
add_test(NAME adaptive_sampler COMMAND adaptive_sampler_test)
add_test(NAME anomaly_detector COMMAND anomaly_detector_test)