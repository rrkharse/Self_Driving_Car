import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def gen_double_set(X_train,y_train):
    
    X_train_plus = []
    y_train_plus = []

    for x in range(len(X_train)):
        X_add = None
        y_add = y_train[x]
        if np.random.rand() < 0.6:
            X_add, y_add = augument(X_train[x],y_train[x]) # With a 60% change
                                                           # Apply the image processing ops 
                                                           # To generate varied data 
        else:
            X_add = mpimg.imread(X_train[x][0])
        X_add = preprocess(X_add)
        X_train_plus.append(X_add)
        y_train_plus.append(y_add)

    return np.array(X_train_plus), np.array([[i] for i in y_train_plus])


def augument(images, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(images, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def preprocess(image):
    image = image[60:-25, :, :] # crop image and remove the sky and the car front
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA) # resize image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) # rgb to yuv
    return image


def choose_image(images, steering_angle):
    choice = np.random.choice(3)
    if choice == 1:
        return mpimg.imread(images[1]), steering_angle + 0.2 # Left image
    elif choice == 2:
        return mpimg.imread(images[2]), steering_angle - 0.2 # Right image
    return mpimg.imread(images[0]), steering_angle # Center image


def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    # (x1, y1) and (x2, y2) to form line between two points
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    # load tensor with same dimensions as input image
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)










