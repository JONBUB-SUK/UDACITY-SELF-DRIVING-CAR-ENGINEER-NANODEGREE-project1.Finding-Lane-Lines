# UDACITY-SELF-DRIVING-CAR-NANODEGREE-project1.-Finding-Lane-Lines

# UDACITY-SELF-DRIVING-CAR-NANODEGREE-project2.-Advanced-Lane-Finding


[//]: # (Image References)

[image1-1]: ./images/1.1_Percaptron,Lambda.JPG "RESULT1"


# Introduction

The object of the project is finding lanes at driving car videos

There are some conditions to pass the project

1. Use Canny edge detection

2. Use Hough transform

3. Draw straight lines on detected lines

4. Apply final function to video



# Background Learning

### Computer vision fundamentals

- Color selection : select using threshold

- Gaussian blur

- Region masking

- Canny edge detection

- Hough transform





# Approach

### 1. Make image to gray scale

It is much more effective using gray scale image rather than RGB channel

#### 1. Import libraries I need

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
%matplotlib inline
```


#### 2. Define grayscale function : return gray scale of input image

```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

(이미지 : gray scale)


### 2. Apply Gaussian blur

#### 1. Define gaussian_blur function : return blurred image

```python
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

(이미지 : after blur)


### 3. Apply Canny edge detection

#### 1. Define canny function : return Canny edge detected region

```python
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```

(이미지 : canny edge detection)


### 4. Apply region masking

Only keeps the region of the image defined by the polygon
ormed from `vertices`. The rest of the image is set to black.
`vertices` should be a numpy array of integer points.

#### 1. Define region_of_interest function : return masked image

```python
def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image
```

(이미지 : masked image)



### 5. Apply Hough transform

Using Hough transform, draw lines on image

#### 1. Find lanes on Canny edge detected image using hough transform

```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    
    return line_img
```

#### 2. Define draw_lines function : connect detected lines

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
      
    ### draw lines betwean both end points ###
    right_line = []
    left_line = []
    
    right_x = []
    right_y = []
    left_x = []
    left_y = []
    

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 > 480:
                right_line.append(line)
            else:
                left_line.append(line)
                
    for line in right_line:
        for x1,y1,x2,y2 in line:
            right_x.append(x1)
            right_x.append(x2)
            right_y.append(y1)
            right_y.append(y2)
    right_x_min = min(right_x)
    right_x_max = max(right_x)
    right_y_min = min(right_y)
    right_y_max = max(right_y)
    
    right_x1, right_y1, right_x2, right_y2 = cal_line_end_point_right(right_x_min, right_y_min, right_x_max, right_y_max)
       
    for line in left_line:
        for x1,y1,x2,y2 in line:
            left_x.append(x1)
            left_x.append(x2)
            left_y.append(y1)
            left_y.append(y2)
    left_x_min = min(left_x)
    left_x_max = max(left_x)
    left_y_min = min(left_y)
    left_y_max = max(left_y)          
    
    left_x1, left_y1, left_x2, left_y2 = cal_line_end_point_left(left_x_min, left_y_max, left_x_max, left_y_min)
    
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)    
```

#### 3. Define cal_line_end_point_left/right function : to extend lines to end point of image


```python
def cal_line_end_point_left(x1,y1,x2,y2):
    m = (y1-y2)/(x1-x2)
    if m != 0:
        x1_end = int((540 + m*x1 - y1)/m)
        y1_end = 540
        x2_end = x2
        y2_end = y2
    else:
        x1_end = 0
        y1_end = y1
        x2_end = x2
        y2_end = y2        
    
    return x1_end,y1_end,x2_end,y2_end

def cal_line_end_point_right(x1,y1,x2,y2):
    m = (y1-y2)/(x1-x2)
    if m != 0:
        x1_end = x1
        y1_end = y1
        x2_end = int((540 + m*x1 - y1)/m)
        y2_end = 540
    else:
        x1_end = x1
        y1_end = y1
        x2_end = 960
        y2_end = y2
    
    return x1_end,y1_end,x2_end,y2_end
```

(이미지 : after Hough transform)


### 6. Draw lines on original image

```python
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)
```


(이미지 : final image)



### 7. Define process_image function : combine every function to apply video

```python
def process_image(image):
    
    image_gray = grayscale(image)
    image_blur = gaussian_blur(image_gray, kernel_size)
    image_canny = canny(image_blur, low_threshold, high_threshold)
    image_masking = region_of_interest(image_canny, vertices)
    image_hough = hough_lines(image_masking, rho, theta, threshold, min_line_len, max_line_gap)
    image_weighted = weighted_img(image_hough, image, α=0.8, β=1., γ=0.)    
    

    return image_weighted
```


### 8. Make video

```python
white_output = 'test_videos_output/solidWhiteRight.mp4'

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```


### 9. Output video

```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```


# Results

This is gif images of final result


(gif 이미지 : 최종 결과 짤)


# Conclusion & Discussion

### 1. About combination of gradient threshold











