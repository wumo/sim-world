#!/bin/bash
# Scripts to build and install native C++ libraries

set -eu

which cmake3 &> /dev/null && CMAKE3="cmake3" || CMAKE3="cmake"
[[ -z ${CMAKE:-} ]] && CMAKE=$CMAKE3
[[ -z ${MAKEJ:-} ]] && MAKEJ=4
[[ -z ${OLDCC:-} ]] && OLDCC="gcc"
[[ -z ${OLDCXX:-} ]] && OLDCXX="g++"
[[ -z ${OLDFC:-} ]] && OLDFC="gfortran"

KERNEL=(`uname -s | tr [A-Z] [a-z]`)
ARCH=(`uname -m | tr [A-Z] [a-z]`)
case $KERNEL in
    darwin)
        OS=macosx
        ;;
    mingw32*)
        OS=windows
        KERNEL=windows
        ARCH=x86
        ;;
    mingw64*)
        OS=windows
        KERNEL=windows
        ARCH=x86_64
        ;;
    *)
        OS=$KERNEL
        ;;
esac
case $ARCH in
    arm*)
        ARCH=arm
        ;;
    aarch64*)
        ARCH=arm64
        ;;
    i386|i486|i586|i686)
        ARCH=x86
        ;;
    amd64|x86-64)
        ARCH=x86_64
        ;;
esac
PLATFORM=$OS-$ARCH
EXTENSION=
echo "Detected platform \"$PLATFORM\""

mkdir -p $PLATFORM
cd $PLATFORM
mkdir -p include lib cache
cd cache

case $PLATFORM in
    linux-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE ../.. -DCMAKE_INSTALL_PREFIX=..
        make -j $MAKEJ
        make install
        ;;
    linux-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE ../.. -DCMAKE_INSTALL_PREFIX=..
        make -n -j $MAKEJ
        make install
        ;;
    macosx-*)
        $CMAKE -DCMAKE_INSTALL_PREFIX=$PLATFORM
        make -j $MAKEJ
        make install
        ;;
    windows-x86)
        CC="gcc -m32" CXX="g++ -m32" $CMAKE ../.. -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread"
        make -j $MAKEJ
        make install
        ;;
    windows-x86_64)
        CC="gcc -m64" CXX="g++ -m64" $CMAKE ../.. -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread"
        make -j $MAKEJ
        make install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac