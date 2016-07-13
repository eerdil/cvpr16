The implementation was used in:

[1] Ertunc Erdil, Sinan Yildirim, Tolga Tasdizen, Mujdat Cetin, 
“MCMC Shape Sampling for Image Segmentation with Nonparametric Shape Priors”, 
Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas.

Any papers using this code should cite [1] accordingly.

The software has been tested under Matlab R2015a and Microsoft Visual C++ 2010.

After unpacking the file and installing the required libraries,
start Matlab and run the following int the root directory:

>> compile_mex_files

If no errors are reported, you can then run "main_mcmc_shape_sampling.m" in the root directory.

If errors are reported, you probably have problem with your mex compiler. Please make
sure that your mex compiler works properly. 

If you still have problems, you can email me at ertuncerdil@sabanciuniv.edu
I will do my best to help.

As is, the code produces results on MNIST and aircraft datasets. You can create a new folder similar to them to test the algorithm on various data sets. Note that, you may also need to change some parameters in main_mcmc_shape_sampling.m where I tried to comment heavily.

Please also report any bug to ertuncerdil@sabanciuniv.edu
