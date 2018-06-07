#1
24bit

#2
Bilinear interpolation uses the weighted average of the surrounding pixels.
Since the scaling factor is 2 then the each new pixel will be the average of a 2x2 block of pixels.

#3
up right up right 4

#4
angiography

#5
Image convolution is the process whereby you take a filter and move it one pixel at a time over an image to create a new image.
Usally the dimension of the filter is small such as a 3x3 filter and the image is of any size.
When the filter is placed upon an area the weights of the filter are multiplied by the weights on the image when the filter lies.
These values are summed and become the value of a pixel in the new image.

#6
apply the laplacian filter to the image then subtract the result from the orignal image

`Sharpened Image = Image - Laplacian(Image)`

#7
imadjust

It allows for the changing of an images gamma. Neccessary to linearise RGB values of an image being displayed on a monitor.

#8
**Why JPEG operates in frequency domain?**
Humans are very good at discerning different intensities but are very poor at distinguishing differences in high frequency data.
Therefore by compressing in the frequency domain a lot more info can be removed whilst retaining a high level of quality.

**why JPEG is lossy?**
JPEG is a lossy compression method in order to maximize the amount of savings that can be obtained by compression. 
Part of the compression process involves quantizing many parts of the image.
This stage causes data which has been deemed to have a low effect on the final input to be removed.

**How image quality can be steerd?**
Quality can be effected in two main ways for JPEG compression.

The first is the amount you choose to scale the chromacity axis after converting the image into YcRcB.
The axis are usally scaled by a factor of two in both directions. This usally results in an indistinguishable change in quallity, 
whilst reducing the amount of color data stored by 4. This value however can be changed in order to effect quality and filesize.

The second way, and the most dominant is by scaling the quantization tables by a quality factor.
If lower quantization values are used then more of the image data is maintained and thus more of the quality.

