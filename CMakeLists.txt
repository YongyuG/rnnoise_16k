cmake_minimum_required(VERSION 2.8)
project(rnn_gao_new C)
set(rnn_gao_v2 main.c)
set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(include)
#AUX_SOURCE_DIRECTORY(. DIR_SRCS)
ADD_SUBDIRECTORY(src)

add_executable(rnn_gao_new main.c)
target_link_libraries(rnn_gao_new rnnLib -lm)
