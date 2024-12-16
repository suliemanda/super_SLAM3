# Docker container for ORB-SLAM3
Clone this repository
```
git clone https://github.com/suliemanda/ORB_SLAM3_docker.git ORB_SLAM3_docker
```
Clone ORB_SLAM3 repository:
```
cd ORB_SLAM3_docker
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
```
Pull ubuntu:22.04 docker image
```
docker pull ubuntu:22.04
```
Build docker container
```
docker build -t ORB_SLAM3_docker . --build-arg NUM_THREADS=$(nproc)
```
To run the container run the file run_docker.bash
```
chmod +x run_docker.bash
./run_docker.bash
```
Inside container run build_orbslam3.bash
```
chmod +x 
./build_orbslam3.bash
```


