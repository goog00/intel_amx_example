#!/bin/bash

# 定义构建目录
BUILD_DIR="build"

# 如果 build 目录存在，则清空其内容
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"/*
else
    mkdir -p "$BUILD_DIR"
fi

# 进入 build 目录
cd "$BUILD_DIR" || exit

# 使用 cmake 配置项目
cmake ..

# 检查 cmake 是否成功
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

# 使用 make 编译项目
make

# 检查 make 是否成功
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

echo "Build completed successfully!"