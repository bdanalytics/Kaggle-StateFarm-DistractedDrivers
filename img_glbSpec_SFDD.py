glbDataFile = {
            'url': 'https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/',
         'filename': 'imgs.zip',
          'extract': False,
    'trnFoldersPth': 'imgs/train',
    'newFoldersPth': 'imgs/test'    
    }

glbRspClass = ['c' + str(id) for id in xrange(10)]
glbRspClassN = len(glbRspClass)
glbRspClassDesc = {
    'c0': 'normal driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger'
    }

glbImgSz = 32 # Pixel width and height.
glbImgPixelDepth = 255.0  # Number of levels per pixel.
glbImgColor = False

glbObsShuffleSeed = 127

glbPickleFile = {
      'data' : 'data/img_D_SFDD_ImgSz_' + str(glbImgSz) + '.pickle',
    'models' : 'data/img_M_SFDD_ImgSz_' + str(glbImgSz) + '.pickle'
    }
        
print 'imported img_glbSpec_SFDD'