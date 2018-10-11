#!/bin/bash
# Scripts to build and install native C++ libraries
mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib cmake_cache
cd cmake_cache
CMAKE_FILE_PATH=../../../cpp
CMAKE_INSTALL_PATH=../

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE ${CMAKE_FILE_PATH} -DCMAKE_INSTALL_PREFIX=..
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE $CMAKE_FILE_PATH -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PATH
        make -n -j $MAKEJ
        make install
        ;;
    macosx-*)
        $CMAKE  $CMAKE_FILE_PATH -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PATH
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE  $CMAKE_FILE_PATH  -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PATH -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread"
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE  $CMAKE_FILE_PATH  -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PATH -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread"
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac