#!/bin/bash
c++ -O3 -c -march=native -Wall -std=gnu++11 -fPIC $(python3-config --includes) randomNumGenerator.cpp -I/c/modular-boost -lquadmath -o randomNumGenerator.o
c++ -O3 -c -march=native -Wall -std=gnu++11 -fPIC $(python3-config --includes) diffusionCDFBase.cpp -I/c/modular-boost -lquadmath -o diffusionCDFBase.o
c++ -O3 -c -march=native -Wall -std=gnu++11 -fPIC $(python3-config --includes) diffusionPDF.cpp -I/c/modular-boost -lquadmath -o diffusionPDF.o
c++ -O3 -c -march=native -Wall -std=gnu++11 -fPIC $(python3-config --includes) diffusionPositionCDF.cpp -I/c/modular-boost -lquadmath -o diffusionPositionCDF.o
c++ -O3 -c -march=native -Wall -std=gnu++11 -fPIC $(python3-config --includes) diffusionTimeCDF.cpp -I/c/modular-boost -lquadmath -o diffusionTimeCDF.o
c++ -O3 -c -march=native -Wall -std=gnu++11 -fPIC $(python3-config --includes) libDiffusion.cpp -I/c/modular-boost -lquadmath -o libDiffusion.o -I"../pybind11/include"

g++ -shared -o "libDiffusion.so" libDiffusion.o diffusionCDFBase.o diffusionPDF.o diffusionPositionCDF.o diffusionTimeCDF.o randomNumGenerator.o -I/c/modular-boost -lquadmath -I"../pybind11/include" $(python3-config --includes)