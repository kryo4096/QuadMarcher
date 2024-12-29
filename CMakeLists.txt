set(CMAKE_MINIMUM_REQUIRED_VERSION 3.25)

project(QuadMarcher)

set(CMAKE_CXX_STANDARD 26)

add_executable(main src/main.cpp)

file(GLOB_RECURSE HEADERS "src/*.hpp")

target_include_directories(main PRIVATE src)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(OpenMP)

target_link_libraries(main PRIVATE sfml-graphics sfml-window sfml-system OpenMP::OpenMP_CXX)

