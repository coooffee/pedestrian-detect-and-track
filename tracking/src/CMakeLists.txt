#PROJECT(deepsort)
#CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#cmake_policy(SET CMP0015 NEW)
#set(tools /usr/bin)
#set(CMAKE_C_COMPILER ${tools}/aarch64-linux-gnu-gcc)
#set(CMAKE_CXX_COMPILER ${tools}/aarch64-linux-gnu-g++)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

AUX_SOURCE_DIRECTORY(./errmsg DIR_SRCS_errmsg)
AUX_SOURCE_DIRECTORY(./thirdPart DIR_SRCS_hungarianoper)
AUX_SOURCE_DIRECTORY(./thirdPart/munkres DIR_SRCS_munkres)
AUX_SOURCE_DIRECTORY(./thirdPart/munkres/adapters DIR_SRCS_munkres_adapters)
AUX_SOURCE_DIRECTORY(./matching DIR_SRCS_matching)
AUX_SOURCE_DIRECTORY(./feature DIR_SRCS_feature)
AUX_SOURCE_DIRECTORY(./common DIR_SRCS_common)
AUX_SOURCE_DIRECTORY(./api DIR_SRCS_api)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

include_directories(
./errmsg
./thirdPart
./thirdPart/Eigen
./thirdPart/munkres
./thirdPart/munkres/adapters
./matching
./feature
./common
./api
)

add_library(deepsort
${DIR_SRCS_errmsg} 
${DIR_SRCS_hungarianoper} 
${DIR_SRCS_munkres} 
${DIR_SRCS_munkres_adapters} 
${DIR_SRCS_feature} 
${DIR_SRCS_matching} 
${DIR_SRCS_api}
)


#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX11)
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else()
#     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()

target_link_libraries(deepsort PUBLIC ${OpenCV_LIBS})
