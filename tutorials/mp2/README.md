# MP2 Camera Calibration

Cameras are used to take pictures of 3D scene. 
The mathematical relationship between the 3D point to its 2D projection on image plane
can be described by a [pinhole model](https://en.wikipedia.org/wiki/Pinhole_camera_model).
For an ideal pinhole model, the relationship can be defined by focal length and principal points. 
When taking geometric distortion into consideration, there will be more parameters to describe the projection.
All these parameters are called **Intrinsic parameters** of cameras.  

The Intrinsic parameters can only describe projection when the world coordinate system is aligned with camera.
However, when the world coordinate system is defined elsewhere, **Extrinsic parameters** need to be determined.
Basically, extrinsic parameters contains a rotation matrix and a translation vector that transforms world coordinates
to camera coordinates.

What camera calibration does is to determine both Intrinsic and Extrinsic parameters of cameras. 
For more details, please read [this](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) 
and [this](https://learnopencv.com/camera-calibration-using-opencv/).

In VPTV, it is important to correctly perform camera calibration. For a particle image, the pixel radius is usually 2 - 4 pixels. 
Without careful calibration, the reprojection error could be as large as 2 pixels, which significantly increase uncertainty of particle centroid detection.


## Calibration board

The first step of camera calibration is to find a calibration board. We usually use 
[Circle Grid](https://github.com/opencv/opencv/blob/3.4/doc/acircles_pattern.png) or 
[Chessboard board](https://github.com/opencv/opencv/blob/3.4/doc/pattern.png) as calibration patterns.
These patterns can be printed on A4 paper with 100% size. However, when the measurement volume is small, 
the patterns can be printed with smaller scale. One thing to remember when using smaller scale is to modify 
the grid size in CirclesGrid class in "calib.hpp".

## Calibration process

For the calibration process, I've made a [video](https://youtu.be/cGvOO-eetoc) showing all processes.

## TODO

1. Finish a calibration process by yourself and save the calibrated files.
2. Test the calibration by reconstructing a board with reflection beads.
3. Write down the detailed calibration process for documentation. 
