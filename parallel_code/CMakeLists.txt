cmake_minimum_required (VERSION 2.6)

SET( SRC_FILES main.c demosaic.c )
ADD_EXECUTABLE( mainBB ${SRC_FILES} )

SET(CMAKE_CFLAGS "${CMAKE_CFLAGS} -std=c99 -O3 -Wall")
MESSAGE("aarch64 build setup")
include_directories("/home/sooyeon/bayer/stub/include")
SET(CMAKE_CFLAGS "${CMAKE_CFLAGS} -DCL_TARGET_OPENCL_VERSION=120")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -L/home/sooyeon/bayer/stub/opencl-1.2-stubs -lOpenCL")

#Set your ndk path
SET(NDK_PATH "/home/sooyeon/ndk_env/aarch64-linux-android-ndk-r18b/bin")
SET(CMAKE_C_COMPILER ${NDK_PATH}/aarch64-linux-android-gcc)
SET(CMAKE_CXX_COMPILER ${NDK_PATH}/aarch64-linux-android-g++)
SET(CMAKE_LINKER ${NDK_PATH}/aarch64-linux-android-ld)
SET(CMAKE_NM ${NDK_PATH}/aarch64-linux-android-nm)
SET(CMAKE_OBJCOPY ${NDK_PATH}/aarch64-linux-android-objcopy)
SET(CMAKE_OBJDUMP ${NDK_PATH}/aarch64-linux-android-objdump)
SET(CMAKE_RANLIB ${NDK_PATH}/aarch64-linux-android-ranlib)

#add_subdirectory
