cd /workspace/ORB_SLAM3
cd Thirdparty/DBoW2 &&\
   mkdir build
   cd build &&\
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 &&\
   make -j
cd ../../g2o &&\
   mkdir build 
   cd build &&\
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 &&\
   make -j
cd ../../sophus &&\
   mkdir build 
   cd build &&\
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 &&\
   make -j
cd /workspace/ORB_SLAM3/Vocabulary &&\
    tar -xf ORBvoc.txt.tar.gz &&\
    cd ..

mkdir build 
    cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 &&\
    make -j4



