cmake_minimum_required(VERSION 3.1)
project(librosa)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -mfma")

add_executable(librosa test.cpp)
target_include_directories(librosa PRIVATE ../include)
target_link_libraries(librosa PRIVATE sndfile)
target_link_libraries(librosa PRIVATE samplerate)