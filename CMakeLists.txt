cmake_minimum_required(VERSION 3.11)

project(ply2gltf)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(ply2gltf io.cpp dump.cpp main.cpp)
target_link_libraries(ply2gltf PRIVATE nlohmann_json::nlohmann_json)
