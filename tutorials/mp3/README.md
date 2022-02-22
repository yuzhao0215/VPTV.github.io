# [MP3: Image processing](https://yuzhao0215.github.io/VPTV.github.io/)
This section aims at processing images and extracting centroids of particles based on the 
results from MP1 and MP2.  
  
In MP1, we created a simple pipeline to read images from cameras then display the images 
on specific window. In MP2, we learned how to calibrate cameras and output the camera 
parameters. Part of the calibration parameters are about the distortion of images. 
Therefore, what we need to do is first read the camera distortion parameters then 
un-distort the images from cameras, then find the centroids and display on images.

The idea is not difficult; 
however, this task utilizes multi-thread programming and therefore should be 
carefully tested. 
The tests will include two parts. 
The first is to simultaneously process images from eight cameras at 
the same time using eight threads. 
The second is to extract the 2D coordinates from eight images also 
using eight threads at the same time.

## TODO
1. Implement different image processing such as "Dialation" and "Erosion;
2. Implement multi-threaded image processing, i.e. for each camera, create a single thread for image processing;