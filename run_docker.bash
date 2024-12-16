xhost +local:docker
docker run --rm -it --gpus all --privileged -e DISPLAY=$DISPLAY --cpuset-cpus="0-15" --net=host  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(pwd):/workspace -w /workspace superpoint_slam bash
