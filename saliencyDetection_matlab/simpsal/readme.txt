simpsal.zip
Saliency (MATLAB source code)
http://www.klab.caltech.edu/~harel/share/gbvs.php

by Jonathan Harel
jonharel@gmail.com
California Institute of Technology

If you use this code, please cite it as:
Jonathan Harel, A Saliency Implementation in MATLAB: http://www.klab.caltech.edu/~harel/share/gbvs.php

================================================================================

This is an installation and help file for the saliency map (MATLAB) code here.

What you can do with this code: Compute the standard Itti, Koch, Niebur (PAMI 1998) 
saliency map for an image or frame sequence (i.e. video), and simplications thereof.

This is intended to be a very simple implementation so that it can be easily 
modified.

================================================================================

Fast start-up procedure:

(1) Put this directory somewhere on your disk, then add the directory to your 
    MATLAB path, or change into this directory to call the code here.

(2) Try running my_simpsal_demo

    open my_simpsal_demo.m to see how it works.

    Basically, at the matlab prompt you can call:

     >> map = simpsal(img);
   
    To get a saliency map 'map' on an input image 'img'. To have it resized
    to the resolution of the input image, use this:

     >> mapbig = mat2gray(imresize( map , [ size(img,1) size(img,2) ] ));

================================================================================

 Helpful Notes:

    (1) the first argument to simpsal() can be an image name or image array

     * if the image array is 4 dimensional ( H x W x {1 or 3} x T ), it is possible 
       to compute temporal feature channels including Flicker and Motion.
       One way to use this on a video would be to pass in a frame buffer
       of the last ~15 frames and take map(:,:,end) as the saliency map
       for the current frame.       

    (2) there is an optional, second, parameters argument
      
     * default_pami_param is used if there is no second parameters argument.
       That tries to approximate the original Itti Algorithm as closely as possible.
  
     * default_fast_param is a setting of parameters which allows you to 
       compute saliency maps very quickly. It excludes an orientation channel,
       and map weighting according to peakiness.

    (3) See default_fast_param.m to see what the parameters do.
   
    (4) See getchan.m to understand how features are computed.

===============================================================================

 Problems / Errors / Troubleshooting : 
     
 1.  Binaries are included for Windows, Linux, and Mac OS X intel.

    However, if you run into problems running mexLocalMaximaGBVS, you may have
    to compile mexLocalMaximaGBVS.cc by typing mex('mexLocalMaximaGBVS.cc') 
    at the matlab prompt when you are in this directory. If you have never
    compiled mex files before, you probably have to run 'mex -setup' first.

 2. If you run into problems running prctile, mat2gray, imshow, or imresize.

    Download http://www.klab.caltech.edu/~harel/share/stupidtools.zip

    And put the contents in this directory.

================================================================================
 Copyright Notice:
    parts of mexLocalMaxima.cc taken from
    http://ilab.usc.edu/toolkit/
================================================================================

Revision History

first authored 12/7/2009
Revised 2/19/2008 fixed bug with Gabor filters
Revised 3/18/2010 added attenuateBordersGBVS to orientation call
Revised 1/17/2011 changed boundary condition in padImage.m
Revised 3/15/2011 added my_imresize since not all matlab versions support 4d imresize
Revised 2/8/2013  fixed bug with flicker channel
