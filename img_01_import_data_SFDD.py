
# coding: utf-8

# State Farm Distracted Drivers
# =============
# 
# Prev Exercises: Udacity:DeepLearning:TensorFlow:notMNIST
# 
# Baseline
# ------------
# 
# notMNIST:
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[11]:

import pprint
import sys
print sys.version


# In[12]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


# ## Analytics Specs

# #### This Project

# In[87]:

glbDataFile = {
            'url': 'https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/',
         'filename': 'imgs.zip',
          'extract': False,
    'trnFoldersPth': 'imgs/train',
    'newFoldersPth': 'imgs/test'    
    }

glbRspClass = ['c' + str(id) for id in xrange(10)]
glbRspClassN = len(glbRspClass)
glbImgSz = 32 # Pixel width and height.
glbImgPixelDepth = 255.0  # Number of levels per pixel.
glbImgColor = False

glbObsShuffleSeed = 127

glbPickleFile = 'img_import_data_SFDD_ImgSz_' + str(glbImgSz) + '.pickle'


# In[88]:

#print 'glbDataFile: %s' % (glbDataFile)
print 'glbRspClass: %s' % (glbRspClass)
print 'glbRspClassN: %d' % (glbRspClassN)
print 'glbPickleFile: %s' % (glbPickleFile)


# #### notMNIST

# In[89]:

# glbDataURL = 'http://yaroslavvb.com/upload/notMNIST/'
# glbImgSz = 32


# ### Import Data

# First, we'll download the dataset to our local machine. 

# In[16]:

def maybe_download(url, filename, expected_bytes = None):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists('data/' + filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat('data/' + filename)
  verified = False
  if (expected_bytes == None):
    if (statinfo.st_size > 0):
        verified = True
  else:      
    if (statinfo.st_size == expected_bytes):
        verified = True
    
  if verified:
    print('Found and verified', 'data/' + filename)
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return 'data/' + filename

dataFNm = maybe_download(glbDataFile['url'], glbDataFile['filename'])


# In[17]:

# url = 'http://yaroslavvb.com/upload/notMNIST/'

# def maybe_download(url, filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   if not os.path.exists(filename):
#     filename, _ = urlretrieve(url + filename, filename)
#   statinfo = os.stat(filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     raise Exception(
#       'Failed to verify' + filename + '. Can you get to it with a browser?')
#   return filename

# train_filename = maybe_download('data/notMNIST_large.tar.gz', 247336696)
# test_filename = maybe_download('data/notMNIST_small.tar.gz', 8458043)


# Extract the dataset from the compressed downloaded file(s).

# In[18]:

def extract(filename, num_classes):
  print("Figure out automatically if data needs to be extracted")
  return
    
  tar = tarfile.open(filename)
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  print('Extracting data for %s. This may take a while. Please wait.' % root)
  sys.stdout.flush()
  tar.extractall()
  tar.close()
  # My edits: data_folders needs to be modified for the correct path
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root)) if d != '.DS_Store']
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

if (glbDataFile['extract']):
    train_folders = extract(os.getcwd() + train_filename, glbRspClassN)
    test_folders  = extract(os.getcwd() + test_filename , glbRspClassN)


# notMNINST:  
# Extraction give you a set of directories, labelled A through J.
# The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the obsNewSet 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# ---
# Inspect data
# ---------
# 
# Let's take a peek at some of the data to make sure it looks sensible. 

# In[19]:

from IPython.display import display, Image


# In[33]:

# Display sample train images
trnFoldersPth = os.getcwd() + '/data/' + glbDataFile['trnFoldersPth']
newFoldersPth = os.getcwd() + '/data/' + glbDataFile['newFoldersPth']
# print(trnFoldersPth)
# print(newFoldersPth)

for cls in glbRspClass:
    print 'Class: %s' % (cls)
    clsPth = trnFoldersPth + '/' + cls
    onlyfiles = [f for f in os.listdir(clsPth) if os.path.isfile(os.path.join(clsPth, f))]
    for ix in np.random.randint(0, len(onlyfiles), size = 3):
        print '  %s:' % (onlyfiles[ix])
#         print '    no size spec:'
#         jpgfile = Image(clsPth + '/' + onlyfiles[ix], format = 'jpg')
#         display(jpgfile)
#         print '    glbImgSz:%d' % (glbImgSz)        
        jpgfile = Image(clsPth + '/' + onlyfiles[ix], format = 'jpg', 
                        width = glbImgSz * 4, height = glbImgSz * 4)
        display(jpgfile)        


# #### notMNINST:  
# Each exemplar should be an image of a character A through J rendered in a different font. 

# In[34]:

# Display sample train images
# train_folders_path = '/Users/bbalaji-2012/Documents/Work/Courses/Udacity/DeepLearning/code/tensorflow/examples/udacity/data/notMNIST_large/'
# glbImgSz = 28
# display(Image(train_folders_path + 'A/a2F6b28udHRm.png', \
#               width = glbImgSz * 4, height = glbImgSz * 4))
# display(Image(train_folders_path + 'B/bnVuaS50dGY=.png', \
#               width = glbImgSz * 4, height = glbImgSz * 4))
# display(Image(train_folders_path + 'C/cmlzay50dGY=.png', \
#               width = glbImgSz * 4, height = glbImgSz * 4))


# Now let's load the data in a more manageable format.
# 
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. The labels will be stored into a separate array (notMNINST only: of integers 0 through 9.)
# 
# A few images might not be readable, we'll just skip them.

# In[40]:

trnFolders = os.getcwd() + '/data/' + glbDataFile['trnFoldersPth']
trnFolders = [trnFolders + '/' + cls for cls in glbRspClass]
print 'trnFolders: %s' % (trnFolders)
newFolders = [os.getcwd() + '/data/' + glbDataFile['newFoldersPth']]
print 'newFolders: %s' % (newFolders)


# In[35]:

# data_folders_path = '/Users/bbalaji-2012/Documents/Work/Courses/Udacity/DeepLearning/code/tensorflow/examples/udacity/data/'
# train_folders = [data_folders_path + 'notMNIST_large/' + d \
#                  for d in sorted(os.listdir(data_folders_path + 'notMNIST_large/')) \
#                     if d != '.DS_Store']
# print train_folders
# test_folders  = [data_folders_path + 'notMNIST_small/' + d \
#                  for d in sorted(os.listdir(data_folders_path + 'notMNIST_small/')) \
#                     if d != '.DS_Store']
# print test_folders


# In[44]:

from scipy import misc as spmisc


# In[105]:

def load(data_folders, max_num_images, max_check = True):
  ids = ['' for ix in xrange(max_num_images)]  
  dataset = np.ndarray(
    shape=(max_num_images, glbImgSz, glbImgSz), dtype=np.float32)
  labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
  label_index = 0
  image_index = 0
  for folder in data_folders:
    print(folder)
    for image in os.listdir(folder):
      if max_check and (image_index >= max_num_images):
        raise Exception('More images than expected: %d >= %d' % (
          image_index, max_num_images))
      elif (image_index >= max_num_images):
        break
        
      image_file = os.path.join(folder, image)
      try:
        rsz_image_data = spmisc.imresize(ndimage.imread(image_file, flatten = not glbImgColor), 
                                      (glbImgSz, glbImgSz))
        image_data = (rsz_image_data.astype(float) -
                      glbImgPixelDepth / 2) / glbImgPixelDepth
        if image_data.shape != (glbImgSz, glbImgSz):
          raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        
        ids[image_index] = image
        dataset[image_index, :, :] = image_data
        labels[image_index] = label_index
        
        if  ((image_index >= 100000) and                            (image_index % 200000 == 0)) or             ((image_index >= 10000 ) and (image_index < 100000) and (image_index % 20000  == 0)) or             ((image_index >= 1000  ) and (image_index < 10000 ) and (image_index % 2000   == 0)) or             ((image_index >= 100   ) and (image_index < 1000  ) and (image_index % 200    == 0)) or             ((image_index >= 10    ) and (image_index < 100   ) and (image_index % 20     == 0)) or              (image_index ==  0    ) :
            print '  image_index: %d; %s:' % (image_index, image)
            display(spmisc.toimage(rsz_image_data))
            
        image_index += 1            
      except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    label_index += 1
    
  num_images = image_index
  ids = ids[0:num_images]  
  dataset = dataset[0:num_images, :, :]
  labels = labels[0:num_images]
#   if num_images < min_num_images:
#     raise Exception('Many fewer images than expected: %d < %d' % (
#         num_images, min_num_images))
  print('Identifiers:', len(ids))
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  print('Labels:', labels.shape)
  return ids, dataset, labels

#smpObsTrnIdn, smpObsTrnFtr, smpObsTrnRsp = load(trnFolders, 250, max_check = False); print smpObsTrnIdn[10:15]
glbObsTrnIdn, glbObsTrnFtr, glbObsTrnRsp = load(trnFolders, 22435)


# In[107]:

print glbObsTrnIdn[100:105]
glbObsNewIdn, glbObsNewFtr, glbObsNewRsp = load(newFolders, 79726)


# In[108]:

print glbObsNewIdn[1000:1005]
savObsNewRsp = glbObsNewRsp
glbObsNewRsp[:] = -1
print glbObsNewRsp[1000:1005]


# In[49]:

# def load(data_folders, min_num_images, max_num_images):
#   dataset = np.ndarray(
#     shape=(max_num_images, glbImgSz, glbImgSz), dtype=np.float32)
#   labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
#   label_index = 0
#   image_index = 0
#   for folder in data_folders:
#     print(folder)
#     for image in os.listdir(folder):
#       if image_index >= max_num_images:
#         raise Exception('More images than expected: %d >= %d' % (
#           image_index, max_num_images))
#       image_file = os.path.join(folder, image)
#       try:
#         image_data = (ndimage.imread(image_file).astype(float) -
#                       glbImgPixelDepth / 2) / glbImgPixelDepth
#         if image_data.shape != (glbImgSz, glbImgSz):
#           raise Exception('Unexpected image shape: %s' % str(image_data.shape))
#         dataset[image_index, :, :] = image_data
#         labels[image_index] = label_index
#         image_index += 1
#       except IOError as e:
#         print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
#     label_index += 1
#   num_images = image_index
#   dataset = dataset[0:num_images, :, :]
#   labels = labels[0:num_images]
#   if num_images < min_num_images:
#     raise Exception('Many fewer images than expected: %d < %d' % (
#         num_images, min_num_images))
#   print('Full dataset tensor:', dataset.shape)
#   print('Mean:', np.mean(dataset))
#   print('Standard deviation:', np.std(dataset))
#   print('Labels:', labels.shape)
#   return dataset, labels

# glbObsTrnFtr, glbObsTrnRsp = load(train_folders, 450000, 550000)
# glbObsNewFtr, glbObsNewRsp = load(test_folders, 18000, 20000)


# We expect the data to be balanced across classes. Verify that.

# In[109]:

print 'glbObsTrnRsp class knts: '
print (np.unique(glbObsTrnRsp, return_counts = True))
print 'glbObsNewRsp class knts: '
print (np.unique(glbObsNewRsp, return_counts = True))


# In[110]:

#print type(glbObsTrnRsp); print glbObsTrnRsp.shape; print glbObsTrnRsp[0:10]
# print np.sum(glbObsTrnRsp == 0)
# print np.unique(glbObsTrnRsp)
# print 'train labels freqs: %s' % \
#     ([np.sum(glbObsTrnRsp == thsLabel) for thsLabel in np.unique(glbObsTrnRsp)])


# Save imported data.

# In[112]:

try:
  f = open('data/' + glbPickleFile, 'wb')
  save = {
    'glbObsTrnIdn': glbObsTrnIdn,
    'glbObsTrnFtr': glbObsTrnFtr,
    'glbObsTrnRsp': glbObsTrnRsp,
#     'glbObsVldFtr': glbObsVldFtr,
#     'glbObsVldRsp': glbObsVldRsp,
    'glbObsNewIdn': glbObsNewIdn,
    'glbObsNewFtr': glbObsNewFtr,
    'glbObsNewRsp': glbObsNewRsp,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', glbPickleFile, ':', e)
  raise
    
statinfo = os.stat('data/' + glbPickleFile)
print('Compressed pickle size:', statinfo.st_size)    


# In[133]:

with open('data/' + glbPickleFile, 'rb') as f:
  save = pickle.load(f)
#   train_dataset = save['train_dataset']
#   train_labels = save['train_labels']
#   valid_dataset = save['valid_dataset']
#   valid_labels = save['valid_labels']
  glbObsNewIdn = save['glbObsNewIdn']
  glbObsNewFtr = save['glbObsNewFtr']
  glbObsNewRsp = save['glbObsNewRsp']
#   test_dataset = save['test_dataset']
#   test_labels = save['test_labels']
  del save  # hint to help gc free up memory
#   print('Training set', train_dataset.shape, train_labels.shape)
#   print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('New set:', len(glbObsNewIdn), glbObsNewFtr.shape, glbObsNewRsp.shape)


# ---
# Inspect Resized Image Data
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. 

# In[114]:

def mydisplayImages(obsIdn, obsFtr, obsRsp):
    imgIxLst = np.random.random_integers(0, obsFtr.shape[0] - 1, 10)
    for imgIx in imgIxLst:
        if (obsRsp[imgIx] > -1):
            print '  imgIx: %d; id: %s; label: %s' %                 (imgIx, obsIdn[imgIx], glbRspClass[obsRsp[imgIx]])
        else:    
            print '  imgIx: %d; id: %s; label: None' % (imgIx, obsIdn[imgIx])    
        plt.figure
        plt.imshow(obsFtr[imgIx,:,:], cmap = plt.cm.gray)
        plt.show()


# In[116]:

print 'Trn set:'; mydisplayImages(glbObsTrnIdn, glbObsTrnFtr, glbObsTrnRsp)


# In[59]:

# dspLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# print 'train set:'
# imgIxLst = np.random.random_integers(0, glbObsTrnFtr.shape[0] - 1, 10)
# for imgIx in imgIxLst:
#     print 'imgIx: %d: label: %s' % (imgIx, dspLabels[glbObsTrnRsp[imgIx]])
#     plt.figure
#     plt.imshow(glbObsTrnFtr[imgIx,:,:], cmap = plt.cm.gray)
#     plt.show()


# In[117]:

print 'New set:'; mydisplayImages(glbObsNewIdn, glbObsNewFtr, glbObsNewRsp)


# ### Shuffle data
# 
# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[129]:

# print type(glbObsTrnIdn)
# smpObsTrnIdn = glbObsTrnIdn[0:4]
# print smpObsTrnIdn
# print [smpObsTrnIdn[ix] for ix in [3, 1, 2, 0]]
# smpObsTrnIdn = [smpObsTrnIdn[ix] for ix in [3, 1, 2, 0]]
# print smpObsTrnIdn


# In[130]:

np.random.seed(glbObsShuffleSeed)
def randomize(ids, dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_ids = [ids[ix] for ix in permutation]
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_ids, shuffled_dataset, shuffled_labels

glbObsTrnIdn, glbObsTrnFtr, glbObsTrnRsp = randomize(glbObsTrnIdn, glbObsTrnFtr, glbObsTrnRsp)
#glbObsNewIdn, glbObsNewFtr, glbObsNewRsp = randomize(glbObsNewIdn, glbObsNewFtr, glbObsNewRsp)


# In[60]:

# np.random.seed(133)
# def randomize(dataset, labels):
#   permutation = np.random.permutation(labels.shape[0])
#   shuffled_dataset = dataset[permutation,:,:]
#   shuffled_labels = labels[permutation]
#   return shuffled_dataset, shuffled_labels
# glbObsTrnFtr, glbObsTrnRsp = randomize(glbObsTrnFtr, glbObsTrnRsp)
# glbObsNewFtr, glbObsNewRsp = randomize(glbObsNewFtr, glbObsNewRsp)


# Check if data is still good after shuffling!

# In[132]:

print 'shuffled Trn set:'; mydisplayImages(glbObsTrnIdn, glbObsTrnFtr, glbObsTrnRsp)
#print 'shuffled New set:'; mydisplayImages(glbObsNewIdn, glbObsNewFtr, glbObsNewRsp)


# Prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune obsTrnN as needed.
# 
# Also create a validation dataset for hyperparameter tuning.

# In[137]:

obsTrnN = glbObsTrnFtr.shape[0] # or fixed number e.g. 20000
obsVldN = int(obsTrnN * 0.2)
print 'obsTrnN: %d; obsVldN: %d' % (obsTrnN, obsVldN)

glbObsVldIdn = glbObsTrnIdn[:obsVldN]
glbObsVldFtr = glbObsTrnFtr[:obsVldN,:,:]
glbObsVldRsp = glbObsTrnRsp[:obsVldN]

glbObsFitIdn = glbObsTrnIdn[obsVldN:obsVldN+obsTrnN]
glbObsFitFtr = glbObsTrnFtr[obsVldN:obsVldN+obsTrnN,:,:]
glbObsFitRsp = glbObsTrnRsp[obsVldN:obsVldN+obsTrnN]

print('   Fitting:', len(glbObsFitIdn), glbObsFitFtr.shape, glbObsFitRsp.shape)
print('Validation:', len(glbObsVldIdn), glbObsVldFtr.shape, glbObsVldRsp.shape)


# In[71]:

# obsTrnN = glbObsTrnFtr.shape[0]
# #obsTrnN = 200000
# obsVldN = 10000

# glbObsVldFtr = glbObsTrnFtr[:obsVldN,:,:]
# glbObsVldRsp = glbObsTrnRsp[:obsVldN]
# glbObsTrnFtr = glbObsTrnFtr[obsVldN:obsVldN+obsTrnN,:,:]
# glbObsTrnRsp = glbObsTrnRsp[obsVldN:obsVldN+obsTrnN]
# print('Training', glbObsTrnFtr.shape, glbObsTrnRsp.shape)
# print('Validation', glbObsVldFtr.shape, glbObsVldRsp.shape)


# Finally, let's save the data for later reuse:  
# Remember to save previous pickled file as '_unshuffled'

# In[75]:

# glbPickleFile = os.getcwd() + '/data/notMNIST.pickle'
# print glbPickleFile


# In[138]:

try:
  f = open('data/' + glbPickleFile, 'wb')
  save = {
    'glbObsTrnIdn': glbObsTrnIdn,
    'glbObsTrnFtr': glbObsTrnFtr,
    'glbObsTrnRsp': glbObsTrnRsp,
        
    'glbObsFitIdn': glbObsFitIdn,        
    'glbObsFitFtr': glbObsFitFtr,
    'glbObsFitRsp': glbObsFitRsp,
        
    'glbObsVldIdn': glbObsVldIdn,        
    'glbObsVldFtr': glbObsVldFtr,
    'glbObsVldRsp': glbObsVldRsp,
        
    'glbObsNewIdn': glbObsNewIdn,        
    'glbObsNewFtr': glbObsNewFtr,
    'glbObsNewRsp': glbObsNewRsp,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', glbPickleFile, ':', e)
  raise
    
statinfo = os.stat('data/' + glbPickleFile)
print('Compressed pickle size:', statinfo.st_size)       


# In[76]:

# #glbPickleFile = 'notMNIST.pickle'

# try:
#   f = open(glbPickleFile, 'wb')
#   save = {
#     'glbObsTrnFtr': glbObsTrnFtr,
#     'glbObsTrnRsp': glbObsTrnRsp,
#     'glbObsVldFtr': glbObsVldFtr,
#     'glbObsVldRsp': glbObsVldRsp,
#     'glbObsNewFtr': glbObsNewFtr,
#     'glbObsNewRsp': glbObsNewRsp,
#     }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#   print('Unable to save data to', glbPickleFile, ':', e)
#   raise


# ---
# Inspect overlap
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

# In[94]:

# print glbObsTrnFtr[0:3]
# print np.ascontiguousarray(glbObsTrnFtr[0:3])
# print np.ascontiguousarray(glbObsTrnFtr[0:3]).shape


# In[139]:

obsFitSet = set(img.tostring() for img in glbObsFitFtr)
print 'Fit: shape: %s vs. len(set): %d pctDups: %0.4f' %     (glbObsFitFtr.shape, len(obsFitSet),      (glbObsFitFtr.shape[0] * 1.0 / len(obsFitSet) - 1) * 100)

obsVldSet = set(img.tostring() for img in glbObsVldFtr)
print 'Vld: shape: %s vs. len(set): %d pctDups: %0.4f' %     (glbObsVldFtr.shape, len(obsVldSet),      (glbObsVldFtr.shape[0] * 1.0 / len(obsVldSet) - 1) * 100)

obsNewSet = set(img.tostring() for img in glbObsNewFtr)
print 'New: shape: %s vs. len(set): %d pctDups: %0.4f' %     (glbObsNewFtr.shape, len(obsNewSet),      (glbObsNewFtr.shape[0] * 1.0 / len(obsNewSet) - 1) * 100) 


# In[79]:

#print glbObsTrnFtr[0:3]
# obsFitSet = set(img.tostring() for img in glbObsTrnFtr)
# print 'train: shape: %s vs. len(set): %d pctDups: %0.4f' % \
#     (glbObsTrnFtr.shape, len(obsFitSet), \
#      (glbObsTrnFtr.shape[0] * 1.0 / len(obsFitSet) - 1) * 100)

# validSet = set(img.tostring() for img in glbObsVldFtr)
# print 'valid: shape: %s vs. len(set): %d pctDups: %0.4f' % \
#     (glbObsVldFtr.shape, len(validSet), \
#      (glbObsVldFtr.shape[0] * 1.0 / len(validSet) - 1) * 100)

# obsNewSet = set(img.tostring() for img in glbObsNewFtr)
# print 'test : shape: %s vs. len(set): %d pctDups: %0.4f' % \
#     (glbObsNewFtr.shape, len(obsNewSet), \
#      (glbObsNewFtr.shape[0] * 1.0 / len(obsNewSet) - 1) * 100)    


# In[142]:

print 'Vld set overlap with Fit set: %0.4f' %     (len(obsVldSet.intersection(obsFitSet)) * 1.0 / len(obsVldSet))
print 'Vld set overlap with New set: %0.4f' %     (len(obsVldSet.intersection(obsNewSet)) * 1.0 / len(obsNewSet))
print 'Fit set overlap with New set: %0.4f' %     (len(obsFitSet.intersection(obsNewSet)) * 1.0 / len(obsFitSet))
# print ' test set overlap with train set: %0.4f' % \
#     (len( obsNewSet.intersection(obsFitSet)) * 1.0 / len( obsNewSet))    
# print 'valid set overlap with  test set: %0.4f' % \
#     (len(validSet.intersection( obsNewSet)) * 1.0 / len(validSet))


# ---
# Stop here!
# ---------
# 
# Following code is in img_02_fit_lgtRgr_SFDD
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# 
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# 
# Optional question: train an off-the-shelf model on all the data!
# 
# ---

# In[110]:

# import graphlab
# print graphlab.version
# graphlab.canvas.set_target('ipynb')


# In[ ]:

# graphlab.logistic_classifier.create(image_train,target='label',
#                                               features=['image_array'])


# In[113]:


print glbObsTrnFtr[0:3,:,:]
print np.reshape(glbObsTrnFtr[0:3,:,:], (3, glbObsTrnFtr.shape[1] * glbObsTrnFtr.shape[2]))
print np.reshape(glbObsTrnFtr[0:3,:,:], (3, glbObsTrnFtr.shape[1] * glbObsTrnFtr.shape[2])).shape


# In[134]:

from sklearn import metrics, linear_model
import pandas as pd


# In[171]:

def fitMdl(nFitObs = 50):
    mdl = linear_model.LogisticRegression(verbose = 1)
    mdl.fit(np.reshape(glbObsTrnFtr[0:nFitObs,:,:],                             (nFitObs, glbObsTrnFtr.shape[1] * glbObsTrnFtr.shape[2])),                  glbObsTrnRsp[0:nFitObs])
    print mdl.get_params()
    print mdl.coef_.shape
    print '  coeff stats:'
    for lblIx in xrange(len(dspLabels)):
        print '  label:%s; minCoeff:row:%2d, col:%2d, value:%0.4f; maxCoeff:row:%2d, col:%2d, value:%0.4f;' %             (dspLabels[lblIx],              mdl.coef_[lblIx,:].argmin() / glbImgSz,              mdl.coef_[lblIx,:].argmin() % glbImgSz,              mdl.coef_[lblIx,:].min(),              mdl.coef_[lblIx,:].argmax() / glbImgSz,              mdl.coef_[lblIx,:].argmax() % glbImgSz,              mdl.coef_[lblIx,:].max())

    train_pred_labels = mdl.predict(np.reshape(glbObsTrnFtr[0:nFitObs,:,:],                                                     (nFitObs               , glbImgSz ** 2)))
    accuracy_train = metrics.accuracy_score(train_pred_labels, glbObsTrnRsp[0:nFitObs])
    print '  accuracy train:%0.4f' % (accuracy_train)
    print metrics.confusion_matrix(glbObsTrnRsp[0:nFitObs], train_pred_labels)

    valid_pred_labels = mdl.predict(np.reshape(glbObsVldFtr,                                                     (glbObsVldFtr.shape[0], glbImgSz ** 2)))
    accuracy_valid = metrics.accuracy_score(valid_pred_labels, glbObsVldRsp)
    print '  accuracy valid:%0.4f' % (accuracy_valid)
    print metrics.confusion_matrix(glbObsVldRsp           , valid_pred_labels)

    test_pred_labels  = mdl.predict(np.reshape(glbObsNewFtr,                                                     (glbObsNewFtr.shape[0], glbImgSz ** 2)))
    accuracy_test = metrics.accuracy_score( test_pred_labels,  glbObsNewRsp)
    print '  accuracy  test:%0.4f' % (accuracy_test)
    test_conf = pd.DataFrame(metrics.confusion_matrix( glbObsNewRsp,  test_pred_labels),                              index = dspLabels, columns = dspLabels)
    print test_conf
    
    return(mdl, (accuracy_train, accuracy_valid, accuracy_test))


# In[172]:

mdl50 = fitMdl(nFitObs = 50) 


# In[181]:

models = pd.DataFrame({'nFitObs': [1e2, 1e3, 1e4, 1e5, glbObsTrnFtr.shape[0]]})
models = models.set_index(models['nFitObs'])
models['mdl'] = linear_model.LogisticRegression()
models['accuracy.fit'] = -1; models['accuracy.vld'] = -1; models['accuracy.new'] = -1

for thsN in models['nFitObs']: 
    models.ix[thsN, 'mdl'], (models.ix[thsN, 'accuracy.fit'],                              models.ix[thsN, 'accuracy.vld'],                              models.ix[thsN, 'accuracy.new'],                             ) = fitMdl(nFitObs = thsN)
    
print models


# In[192]:

plt.figure()
plt.plot(models['nFitObs'], models['accuracy.fit'], 'bo-', label = 'fit')
plt.plot(models['nFitObs'], models['accuracy.vld'], 'rs-', label = 'vld')
plt.plot(models['nFitObs'], models['accuracy.new'], 'gp-', label = 'new')
plt.legend()
plt.title("Accuracy")
plt.xscale('log')
axes = plt.gca()
axes.set_xlabel('nFitObs')
# axes.set_xlim([mdlDF['l1_penalty'][mdlDF['RSS.vld'].argmin()] / 10 ** 2, \
#                mdlDF['l1_penalty'][mdlDF['RSS.vld'].argmin()] * 10 ** 2])
# axes.set_ylim([0, mdlDF['RSS.vld'].min() * 1.5])
plt.show()


# In[ ]:




# In[ ]:




# In[123]:

print dspLabels


# In[154]:

import pandas as pd


# In[ ]:



