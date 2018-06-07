1.
24bit

2.
Bilinear interpolation uses the weighted average of the surrounding pixels.
Since the scaling factor is 2 then the each new pixel will be the average of a 2x2 block of pixels.

3.
up right up right 4

4.
angiography

5.
Image convolution is the process whereby you take a filter and move it one pixel at a time over an image to create a new image.
Usally the dimension of the filter is small such as a 3x3 filter and the image is of any size.
When the filter is placed upon an area the weights of the filter are multiplied by the weights on the image when the filter lies.
These values are summed and become the value of a pixel in the new image.

6.
apply the laplacian filter to the image then subtract the result from the orignal image
`Sharpened Image = Image - Laplacian(Image)`

7.
imadjust

It allows for the changing of an images gamma. Neccessary to linearise RGB values of an image being displayed on a monitor.

8.
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

9.
**3 Main approaches to visualise image databases**
Heirachical Based
Images are aranged in some sort of heirerachy such as a tree.
Users are presented with a representative image which summarizes an entire section of the database. They then have the ability to explore that image. Then they will be shown the images below that main image. Each image shown here is also representative of a further set of images.

Clustering Based

Mapping Based
Using features extracted from the images they are aranged onto a single surface in either 2d or 3d. Images close to each other are similar to one another. For example you may map images based on the average color of the image, placing blue images at one end of the surface, transitioning to yellow on the other side. Then all images can be placed somewhere on the spectrum creating a smooth range.

10.
**Run length coding** works by replacing repeating characters with a single instance of the character and the amount of times it should be repeated. This is most effective on data with lots of repition and can actually cause an increase in size if there is no repetion and/or the string is very short.

**Huffman coding** works by finding the frequency of elements in a given string. Then assigns codes to each element with more frequent elements getting assigned smaller codes and less frequent ones assigned longer codes. Huffman coding provably offers the most efficient lossless storage of data.

One detriment to Huffman coding is the need to store the table which converts elements into codes. The need to store this table means Huffman coding can be ineffective when compressing small strings. However the size of the table usally grows much slower then the data itself. If forinstance the data only contained unique elements then huffman coding would simply add a layer of abstraction upon the data without any compression and with the added need to store a conversion table.

11.
**K-means ckustering** works by:
1. selecting the number of clusters being used.
2. choosing centers for each cluster randomly or otherwise
3. assigning each element to be clustered to the nearest cluster.
4. adjust the center of the cluster to minimise the distance to all assigned elements.
5. repeat steps 2-4 until no change occurs

This method can be used for image segmentaion with no modifications. One challanging thing with using k-means for this task is chosing the number of clusters to use. As it is hard to know how many objects there are to be distinguished prior to actually segmenting the image.

12.
A histogram can be extracted by 
