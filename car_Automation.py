

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pdZ
# path last  part cut
import ntpath
import random
data_directory = 'C:/Users/Ananthu K S/Desktop/DATA'
# a list
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
# load the file (csv)
data = pd.read_csv(os.path.join(data_directory, 'driving_log.csv'), names = columns)
# file name too long(image) so placeholders come ie,... so to avoid  max_colwidth
pd.set_option('display.max_colwidth', -1)
# shows the data..
data.head()
plt.interactive(False)

#cut the last part from path
def path_leaf_cut(path):
    head, tail = ntpath.split(path)
    return tail

#Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series
data['center'] = data['center'].apply(path_leaf_cut)
data['left'] = data['left'].apply(path_leaf_cut)
data['right'] = data['right'].apply(path_leaf_cut)
data.head()

#distribution of diff angles
#bins is for diff angles with in a limit(our case -1  to 1 is divided into 25)

num_bins_histo = 25

samples_per_bin = 400

#what all we get from steering is divied into 25 intervals
#bins intervlas of steering angles

hist, bins = np.histogram(data['steering'], num_bins_histo)
#when we look through the bins we dont have value  0 for straight so we
#slice from beg to end and add to from value at 1  to end elementwise addition
#4th -0.4 and 5th 0.4 so add we get 0 and we do *0.5 to get back to orginal values
center = (bins[:-1]+ bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
print('total data:', len(data))
remove_list = []
#removed samples from  the bin(samples we want to remove )
#looping through every bin
for j in range(num_bins_histo):
  list_angles = []
  #loop through every angles in the bin
  for i in range(len(data['steering'])):
      #checks with the current bin and check with the next one
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_angles.append(i)
  #if a bin has suppose 500  angles we have to remove 100 as our sample bin os 400, we cant remove 100 from first and last or center becasue the data is stored wrt to the track
  #so we shuffle the list
  list_angles = shuffle(list_angles)
  list_angles = list_angles[samples_per_bin:]
  remove_list.extend(list_angles)
 
print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))
 
hist, _ = np.histogram(data['steering'], (num_bins_histo))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))


print(data.iloc[1])
def load_img_steering(data_directory, data_frames):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    #iloc for row based(selection of  data)
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    #strip for eliminating spaces in file string(IMG)
    image_path.append(os.path.join(data_directory, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_path.append(os.path.join(data_directory,left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_path.append(os.path.join(data_directory,right.strip()))
    steering.append(float(indexed_data[3])-0.15)

  #converting to array(asarray)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

 #image and corresponding steering angles
image_paths, steerings = load_img_steering(data_directory + '/IMG', data)

#from sklearn train_test_split

#random for splitting data in an random manner(test valid  -----> split)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

#1 ROW 2 COLS
figure, axis = plt.subplots(1, 2, figsize=(12, 5))
#MORE THAN 1 SUBPLOTS SO WE GET AN ARRAY OF AXES
axis[0].hist(y_train, bins=num_bins_histo, width=0.05, color='blue')
axis[0].set_title('Training set')
axis[1].hist(y_valid, bins=num_bins_histo, width=0.05, color='red')
axis[1].set_title('Validation set')


def zoom(image):
    #affine preserve straight lines within the image
    #scale[range of zoom 1--->1.3(30%)]
  zoom = iaa.Affine(scale=(1, 1.25))
  image = zoom.augment_image(image)
  return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)

figure, axs = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')


def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)

figure, axs = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(panned_image)
axs[1].set_title('Panned Image')
def img_random_brightness(image):
    #multiple pixel intensity
    #brighter is larger than 1 and if <1 darker
    brightness = iaa.Multiply((0.3, 1.4))
    image = brightness.augment_image(image)
    return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)

figure, axs = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(brightness_altered_image)
axs[1].set_title('Brightness altered image ')


def img_random_flip(image, steering_angle):
    #2nd arg: 0-->vertical 1 -->hori-1 -->both
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle
random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
 
 
original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)

figure, axs = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
 
axs[1].imshow(flipped_image)
axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))


def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle
col_num = 2
row_num = 10

figure, axs = plt.subplots(row_num, col_num, figsize=(15, 50))
figure.tight_layout()
 
for i in range(10):
  randnum = random.randint(0, len(image_paths) - 1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]
    
  original_image = mpimg.imread(random_image)
  augmented_image, steering = random_augment(random_image, random_steering)
    
  axs[i][0].imshow(original_image)
  axs[i][0].set_title("Original Image")
  
  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title("Augmented Image")
 
def img_preprocess(img):
    #cropping the image
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    #smaller image(size is used by nvidia model too)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
image = image_paths[75]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

#subplots for org and prepro images
#doing some basic image processing (noise reduction,soothing,smooth-->gaussian blur)
#Nvidia Model so(Color Scheme,space---> RGB2YUV brightness(y)  (uv)-->color)
figure, axs = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')

def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))  
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

figure, axs = plt.subplots(1, 2, figsize=(15, 10))
figure.tight_layout()
 
axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')
 
axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')
def nvidia_model():
  model = Sequential()
  #24 filters (5*5 matrix)   subsample(Stride(2*2))
  #input shape of img

  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
  #dimentions decreases so we removed subsample
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  
  model.add(Convolution2D(64, 3, 3, activation='elu'))
#   model.add(Dropout(0.5))
  
  #converts into one dimentional(fully connected layer)
  model.add(Flatten())

  #100 nodes
  model.add(Dense(100, activation = 'elu'))

  #Use Dropout layers, which will randomly remove certain features by setting them to zero.
  #50% of out to 0
#   model.add(Dropout(0.5))
  
  model.add(Dense(50, activation = 'elu'))
#   model.add(Dropout(0.5))
  
  model.add(Dense(10, activation = 'elu'))
#   model.add(Dropout(0.5))
 #single output(steering angle)
  model.add(Dense(1))
  
  optimizer = Adam(lr=1e-3)
  #compile the model(to train)
  #meansquarederror
  model.compile(loss='mse', optimizer=optimizer)
  return model
model = nvidia_model()
print(model.summary())
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
model.save('model.h5')
from google.colab import files
files.download('model.h5')





