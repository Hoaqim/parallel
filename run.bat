@echo off
set filename=%1
cd build
cmake -B build
cmake --build build
.\build\Debug\%filename%.exe
cd ../