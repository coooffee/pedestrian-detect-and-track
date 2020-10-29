cp -f ./src/api/deepsort.h ./include/
mkdir build && cd build

cmake ../src -DCMAKE_BUILD_TYPE=Debug
make -j4

cd ..
cp build/libdeepsort.so ../


