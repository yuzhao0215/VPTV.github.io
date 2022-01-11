Learn VPTV with six mini projects

# [MP1: Synchronization](https://yuzhao0215.github.io/VPTV.github.io/) 
Synchronization of multiple cameras is the base requirement of VPTV. The synchronization is achieved by camera software development kit (SDK) from our cameras’ manufacture “Optitrack”. Therefore, the first mini project is to master Optitrack camera SDK. This SDK provides functions to control cameras in terms of synchronization, camera frame rate, exposure, and image segmentation methods. A test using stopwatch will be built to test if eight cameras capture images at the same instant. If synchronized correctly, all images will show the same time stamp.

# [MP2: Calibration](https://yuzhao0215.github.io/VPTV.github.io/)
Calibration ensures the 2D and 3D coordinates are accurate. The calibration is implemented using OpenCV package with C++. By finishing this project, one will be familiar with calibrating multiple cameras and store the calibrated parameters of cameras for future use. Several tests will be used using different camera setups from small scale (nozzle flow) to large scale (building model). 

# [MP3: Image processing](https://yuzhao0215.github.io/VPTV.github.io/)
This section is also using OpenCV. The main objectives are binarizing images and extracting centroids. The idea is not difficult; however, this task utilizes multi-thread programming and therefore should be carefully tested. The tests will include two parts. The first is to simultaneously process images from eight cameras at the same time using eight threads. The second is to extract the 2D coordinates from eight images also using eight threads at the same time.

# [MP4: Correspondence](https://yuzhao0215.github.io/VPTV.github.io/)
This is the core part of VPTV, and therefore needs more attention. Unlike other mini projects dependent to libraries such as SDK and OpenCV, there are much more code in this project. In addition, GPU programming is also needed to achieve parallel correspondence.  The tests involve corresponding sample particles using both CPU and GPU. The accuracy of correspondence and operation time will be evaluated.

# [MP5: Temporal tracking](https://yuzhao0215.github.io/VPTV.github.io/)
This part is relatively simple. It can be considered as a nearest neighbor algorithm be simply extending current particle position and search for candidates. There is no parallel programming involved in current VPTV code. The test will compare the difference between predicted and known positions.

# [MP6: Real-time visualization](https://yuzhao0215.github.io/VPTV.github.io/)
Although the name contains visualization, the main objective in this mini project is to learn the data pipeline in real-time VPTV. This pipeline is the most important feature that makes real-time measurement possible. Therefore, this pipeline connects all mini projects from 1-5 and the test will be testing if a real-time visualization of synthetic test can be achieved.