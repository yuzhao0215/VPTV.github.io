# [MP4: Correspondence](https://yuzhao0215.github.io/VPTV.github.io/)
In 3D reconstruction, [correspondence](https://en.wikipedia.org/wiki/Correspondence_problem) identifies which parts of one image correspond to which parts of another 
image with different viewing directions or camera parameters. 
In this mini project, the criterion to determine the spatial correspondence of particles is [epipolar geometry constraint](https://en.wikipedia.org/wiki/Epipolar_geometry). 

A two-stage method was proposed in our lab to get all four-camera corresponded lists. 
First, a matching array with dimensions [N, P, N-1, M] was created. 
Among the dimensions, N was the number of cameras, 
P was an arbitrary number representing the maximum 2D particles in each image, 
and M was an arbitrary number representing the maximum particle candidates along with each epipolar line. 
The last dimension data of the matching array was a matching list that contained the indexes of 
candidate particles in the target camera that could correspond to Particle i in the source camera.

<p align="center">  
<img width="400" height="200" src="./images/mp4/correspondence_array.png">  
</p>

After the matching array was initiated, all possible two-camera correspondence were determined in the first stage. 
A particle pair {X<sub>A,i</sub>, X<sub>B, j</sub> } is called a two-camera correspondence if  
X<sub>B, j</sub><sub>T</sub>∙F<sub>A, B</sub>∙X<sub>A,i</sub> < threshold, 
where A and B are indexes of cameras, i and j are indexes of particles in cameras A and B, and X refers to 
2D particle coordinates in pixels. The threshold is an arbitrary value in pixels. 
The implementation refers to a nested loop that iterates all possible camera pairs {A, B} and all particle 
pairs in these two cameras. 

In the second stage, all particle pairs that satisfied the two-camera epipolar constraint acquired in stage one 
are evaluated to determine if they satisfy the four-camera epipolar constraints. 
A four-camera correspondence {X<sub>A,i</sub>,X<sub>B,j</sub>,X<sub>C,k</sub>,X<sub>D,l</sub> } is determined if it satisfies the following equation.

<p align="center">  
<img width="200" height="220" src="./images/mp4/equation.PNG">  
</p>

## Dataset
A 3D PIV dataset is used in this mini project. In this dataset, six cameras are used to capture tracer particles 
in a jet flow close to a wall. There are 145 framegroups, where each framegroup contains six images. 
The number of particles in each image is approximately 300.

## TODO
1. Understand the code in modules "correspond" and "correspondCUDA";
2. Understand the code in "main.cpp" and make comments on code about what the code is doing;