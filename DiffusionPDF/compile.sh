#!/bin/bash
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) diffusionPDF.cpp -I/c/modular-boost -lquadmath -o diffusionPDF.so -I"../../pybind11/include"
