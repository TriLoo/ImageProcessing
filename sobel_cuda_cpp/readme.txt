this file record the bugs when I optimized the origin version of sobel.cu using 2D texture memory.

Firstly, I met the biggest question is that I cannot read the texture or failed to bind the texture memory or cannot fetch the texture data correctly. 

After google for it, I found that the 2D texture should be bound to a 2D array if you want to use tex2D to fetch the data from it. the usage of 2D array can refer to the <CUDA programming guide>.

Another thing is that we have bind our texture memory to the created 2D array which include the image data, and now we want to fetch them in the kernel function, the correct practice is to set the "normalized coordinates the texture" to be "false", otherwise, we should access the texture via the coordinates from 0 to 1. i.e. float u = x / (float) width; && foat v = y / (float)height; x and y is the index of thread in the x and y direction.


for more details, you can find it in https://stackoverflow.com/questions/10565420/bind-cuda-texture-to-a-float-image/10566322#10566322?newreg=2603e58cba25413b8c6eae6566f409f5   
