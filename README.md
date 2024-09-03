

# Photometric 3D Reconstruction, Pose and Deformation Estimation in GutSLAM
This repository contains the code and resources for the GutSLAM project, which focuses on photometric 3D reconstruction along with pose and deformation estimation techniques. The project utilizes advanced computer vision and photometric analysis to enhance SLAM (Simultaneous Localization and Mapping) applications in complex environments.

## Dependencies
- Python 3.11.7
- NumPy
- OpenCV-Python
- SciPy
- Open3d
- Pyvista
- Matplotlib




### **/ttp_nrsfm**
Directory containing core Python scripts for Non-Rigid Structure from Motion (NRSFM) using Deformable Shape and Motion computation.

- `cproj.py` : Camera projection utilities.
- `mdh_nrsfm_socp.py` : Implementation of NRSFM SOCP.
- `demo_mdh.py` : Demonstrative script for DSM computation methods.
- `main.py` : Main entry script for photometric reconstruction.
### **/gut_images**
Holds sample images and figures utilized within the project.

### **/pose_estim**
Includes scripts and data for pose estimation and optimization.

- **Optimization Scripts**:
    - `S_FrameOpt.py` : Optimizes parameters for single-frame pose estimation.
    - `FramesOpt.py` : Multiple frames optimization for pose and control point estimation.
    - `MultiFramePCPOpt.py` : Joint Optimization of pose and control points across sequential frames.
- **/logs**: Logs output from the optimization processes, including error reports and parameters.
- **/rendering**: Contains rendered meshes and scenes for visualization.
- **/data**: Control points data files for optimization scripts.
### **/photometric_rec/py**
Scripts related to photometric reconstruction components of the project.

- `main.py` : Entry point for photometric reconstruction processes.
- `reconstruct.py` , `calib.py` , `visualize.py` : Utilities for reconstruction, calibration, and visualization.
- `/pcl_output` : Output text files of point cloud data.
- `/image_output` : Output images such as depth maps.





