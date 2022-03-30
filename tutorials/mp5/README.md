# [MP5: Tracking](https://yuzhao0215.github.io/VPTV.github.io/)
This part is relatively simple. It can be considered as a nearest neighbor algorithm be simply extending current particle position and search for candidates. 
There is no parallel programming involved in current VPTV code. The test will compare the difference between predicted and known positions. 
The test dataset is the same as the MP4-correspondence. 
After tracking the trajectories in 145 frames, the tracked trajectories are compared to the ground-truth trajectories. 

## TODO
1. Understand the code in tracking module;
2. Understand the code in "main.cpp" and comment on code;