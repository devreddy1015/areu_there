# Saans sensor node — ESP-IDF v5.4, target esp32s3.
#
# Built through Docker so the toolchain is pinned and reproducible; see
# ../build.sh. Nothing here assumes a locally installed IDF.
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(saans_node)
