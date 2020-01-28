import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

file_list = []
class_list = []

direct = '/path/to/dataset'

cat = os.listdir('/path/to/dataset')

# identifies the subfolders that contain the images we will use
for category in cat:
  path = os.path.join(direct, category)

  for subdir, dirs, files in os.walk(path):
    for file in files:
      filepath = subdir
      
      if filepath.endswith('Environment'):
        end = filepath
      elif filepath.endswith('Net'):
        end = filepath
      elif filepath.endswith('images'):
        end = filepath
      else:
        pass

  for img in os.listdir(end):
    img_array = cv2.imread(os.path.join(end, img), cv2.IMREAD_GRAYSCALE)

train_data = []

#setting the parameters in which the images will be resized to
img_size = 50

# a function to convert image files into data that can be fed into the CNN model
def create_train_data():
  for category in cat:
    path = os.path.join(direct, category)
    class_num = cat.index(category)

    for subdir, dirs, files in os.walk(path):
      for file in files:
        filepath = subdir
        
        if filepath.endswith('Environment'):
          end = filepath
        elif filepath.endswith('Net'):
          end = filepath
        else:
          pass

    # each image in a folder are converted to grayscale and resized to a dimension of img_size by img_size. This image is then appended to the
    # train_data array with their encoded class
    for img in os.listdir(end):
      try:
        img_array = cv2.imread(os.path.join(end, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
        train_data.append([new_array, class_num])
      except Exception as e:
        pass

create_train_data()

#randomizes the order of the training data
random.shuffle(train_data)

X = []
y = []

# identifies the image array as the feature and the encoded class as the label
for features, label in train_data:
  X.append(features)
  y.append(label)

# converts X from a list to an array
X = np.array(X).reshape(-1,img_size, img_size, 1)

# serializes the X and y arrays so that they can be stored for later use by the CNN model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
