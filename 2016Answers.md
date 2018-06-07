1. L = 2 ^ k, k = 8, L = 256

2. Increasing or decreasing the size of an image, techniques include nearest-neighbour and bilinear

3. m-path: u, r, u, r, Length = 4

4. Changing the gamma can made certain details more visable a gamme greater than 1 will bring out highlights in vright areas,
less than 1 will show details in dark areas
   
5. A filter that sets the value of the selected (middle) pixel to be the median value (middle) of the values within the 3x3 filter.
Used to reduce random noise, such as salt and pepper noise, and blurr an image, usually works better than a averaging filter.
   
6. An image histogram can show whether the image contains mainly dark pixels (clustering towards the left), mainly light pixels
(clustering towards the right), whether there is low contrast (pixels are thinly distributed) or high contrast 
(pixels are widely distrbuted)
   
7. Function to create an equalized histogram of an image, useful for compairing images with different illumination etc
   7. Performs historgram equalization on an image `im`.

8.   0 1 2 3
   0 1 1 0 1
   1 0 0 0 0
   2 1 0 1 0
   3 1 0 0 0
   
9. ?

10. Human visual system is good at seeing differences in low frequency inputs, not as good at high frequency therefore high frequency 
data can be compressed, incidently high frequency data can be compressed to easiliy to save space, win - win

11. Need to learn

12. Similar frames will have similiar histograms, similar histograms indicate little change in frames and therefore the scene.
By using a threshold value, a new scene can be detected by looking for a change in histogram values that exceed the threshold value

13. Nope

14. Difficult to put here, but known

15. Erosion - A filter is passed over each pixel of an image, if the pixel does not trigger a fit (all 1's in filter are over an active
pixels), then the pixel is removed from the new image, otherwise it is kept. Used for trimming extrusions

Dilation - A filter is passed over each pixel of an image, if the pixel triggers a hit (atleast one pixel falls under a 1 in the filter)
then the pixels corresponding to the filter are acivated in the new image. Used for filling incursions

Opening - erosion followed by dilation. Used for seperating images that ae joined

Closing - dilation followed by erosion. Used for joining images that are close

Give example images
