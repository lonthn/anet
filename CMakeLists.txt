cmake_minimum_required(VERSION 2.8.12)
project(anet)

#set(CMAKE_CXX_STANDARD 17)
# Google Address sanitizer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories(.)

set(anet_src
        src/layer/convolution_layer.cpp
        src/layer/full_connection_layer.cpp
        src/layer/pooling_layer.cpp
        src/layer/activation_layer.cpp
        src/layer/reshape_layer.cpp
        src/layer/dropout_layer.cpp

        src/math/functions.cpp

        src/util/iobyte.cpp

        src/network.cpp)

set(example
        example/train_mnist.cpp)

link_directories(/usr/local/lib/)

add_executable(mnist
        ${anet_src}
        ${example})
#target_link_libraries(mnist
#        uv_a)

add_library(anet
        ${anet_src})
