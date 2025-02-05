# ORB-SLAM3 with SuperPoint, SuperGlue, and Panorama Camera Support  

This repository extends ORB-SLAM3 by:  
- Replacing ORB features with **SuperPoint** for feature extraction.  
- Using **SuperGlue** for feature matching.  
- Adding support for **panorama (equirectangular) cameras**.  

## Features  
- Improved feature extraction and matching with deep learning-based methods.  
- Enhanced performance in challenging environments where ORB features struggle.  
- Support for wide-angle and 360° images using an equirectangular camera model.  
- **Dockerized setup** for easy installation and reproducibility.  

## Installation & Setup  
1. Build the Docker image:  
   ```bash
   docker build -t ORB_SLAM3_docker . --build-arg NUM_THREADS=$(nproc)
   ```
2. Run the container:  
   ```bash
   ./run_docker.sh
   ```
   This will start the SLAM system inside a container with all dependencies installed.  
3. Build the package:
```bash
./build_orbslam3.bash
```

