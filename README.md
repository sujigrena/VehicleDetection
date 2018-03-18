
## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4
[video2]: ./yolo_output.mp4
[img1]: ./output_images/cars.png
[img2]: ./output_images/noncars.png
[img3]: ./output_images/hog_car_5.png
[img4]: ./output_images/notcars_5.png
[img5]: ./output_images/hog2.png
[img6]: ./output_images/hog4.png
[img7]: ./output_images/ycrcb_16.png
[img8]: ./output_images/ycrcb_cr.png
[img9]: ./output_images/luv_l.png
[img10]: ./output_images/luv_u.png
[img11]: ./output_images/hog1.png
[img12]: ./output_images/hog3.png
[img13]: ./output_images/slidewindow1.png
[img14]: ./output_images/sliding2.png
[img15]: ./output_images/op1.png
[img16]: ./output_images/op3.png
[img17]: ./output_images/op4.png
[img18]: ./output_images/op5.png
[img19]: ./output_images/hotwin.png
[img20]: ./output_images/label.png
[img21]: ./output_images/yoloarch.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

##### Vehicles
![alt text][img1]

##### Non Vehicles
![alt text][img2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

##### Vehicles HOG Sample 
![alt text][img3]

##### Non Vehicles HOG Sample 
![alt text][img4]

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

##### Vehicles HOG
![alt text][img5]

##### Non Vehicles HOG
![alt text][img6]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (orientation,pixels per cell and cells per block) with different color spaces as follows:

* color space= 'YCrCb' orientations=8, pixels_per_cell=(16, 16) and cells_per_block=(2, 2) Channel used= Y.

This combination doesn't seem to reflect the features of the car as expected
![alt text][img7]


* color space= 'YCrCb' orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(4, 4) Channel used = Y

This combination seemed to perform better than the previous combination

![alt text][img8]


* color space= 'LUV' orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) Channel used = U

LUV color space seemed to work better. So I explored the HOG features of different channels in the same color space. U space didn't seem to peform as expected.

![alt text][img10]

* color space= 'LUV' orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) Channel used = L

The L channel produced better results compared to all other combinations.

![alt text][img9]

* color space= 'YUV' orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) Channel used = Y

Then I explored YUV color space and tried Y channel to extract the HOG features. I felt this was the best combination as it reveals the features of both vehicle and non vehicles images distinctly.


##### Vehicles HOG
![alt text][img11]

##### Non Vehicles HOG
![alt text][img12]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVC with rbf kernel and with C value 10. I've used around 8500 images for each car and non-car classes. I used the spatial colour, color histogram as well as hog features of each image as an input. The spatial color was nothing but flattend image in YUV color space. All the three channels of the color space were flattened and taken as first feature. Then I applied histogram to three channels and taken as the second feature. As we know color histogram represents the distribution of the values across each channel. Finally I took the hog feature for the Y channel and added it a feature.

After extracting all the features, I applied normalization using standard scalar. This was very necessary because I have combined totally three different features which could bias the training towards the dominant feature (or feature with maximum magnitude). Also I applied the normalization only for training data and kept the test data untouched.

Then fed the normalized data for training the SVC classifier. Initially I tried to train a LinearSVC model, but it seemed to detect so many false positive values despite getting accuracy of around 98%. Then I swiched to SVC classifier. After training the test accuracy was around 99.3% and the model performs very well and detects very rare false positives.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions where the cars are supposed to appear. I filtered out the unncessary portions at the top where there is zero chances of a car appearing.I used two windows to capture the cars at various positions of the screen. The first window searches the top portion of the car region, searching with small windows as the cars in this region will be relatively smaller. The y regions used is from 400 to 500 and the window size used was 48x48

![alt text][img13]

Then I used another sliding window of size 98x98 to capture cars which appear relatively closer in the image.

![alt text][img14]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][img15]
![alt text][img16]
![alt text][img17]
![alt text][img18]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have used all the possible features of the input image, including spatial binning, color histogram and hog features. The total number of features I used was around 2580 and it took quite a lot of time to extract these many features from 16000 images. The training time was also high for SVC classifier. I've handled false positives with hot maps. I may true to reduce the number of features and hence less feature extraction time and less training time.

Also there is one very minute false positive detected at 0:39 seconds of the video despite adding heat vectors and filtering with thresholds. I may further refine the thresholds or the sliding window region to avoid even a single false postive detection.

#### 2. YOLO implementation:

I've tried to implement the same using YOLO model as well. As I was more interested in DEEP neural network, and seeing the excellent architecture of YOLO, I thought of giving a try.

![alt text][img21]

I have used the pyyolo from https://github.com/digitalbrain79/pyyolo, which the python implementation of YOLO ( which was designed in Darknet). They've given the trained weights to load and detect the classes in the image that we feed, which then predicts the classes, their bounding boxes and probability of the class. I've filtered only car classes, and fixed a probability threshold. The result video is attached.

Its not as smooth as SVM, may be I will enhance in future by fine tuning the heat maps to bring the smooth boundary boxes. But the accuracy is pretty good which could be improved by increasing the threshold I believe.

Here's a [link to my yolo result](./yolo_output.mp4)

