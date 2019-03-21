cmake_minimum_required(VERSION 3.5)
project(nanogui_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

FIND_PACKAGE(Boost) 
MESSAGE("Boost information:") 
MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_subdirectory(lib/nanogui)
include_directories(lib/nanogui/include)
include_directories(${NANOGUI_EXTRA_INCS})
add_definitions(${NANOGUI_EXTRA_DEFS})

include_directories(lib/embed-resource)
add_subdirectory(lib/embed-resource)

embed_resources(MyResources shader/vert.glsl shader/frag.glsl)

add_executable(mcaq test.cpp ${MyResources})
target_link_libraries(mcaq nanogui ${NANOGUI_EXTRA_LIBS})
