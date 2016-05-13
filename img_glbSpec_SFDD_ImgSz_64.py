glbDataFile = {
            'url': 'https://www.kaggle.com/c/state-farm-distracted-driver-detection/download/',
         'filename': 'imgs.zip',
          'extract': False,
    'trnFoldersPth': 'imgs/train',
    'newFoldersPth': 'imgs/test'    
    }
    
glbDataScrub = {
    'trn': {
            'img_15117.jpg': {'rsp': ('c0', 'c9') },
            
            'img_78504.jpg': {'rsp': ('c5', 'c8') },              
            
            'img_73378.jpg': {'rsp': ('c7', 'c9') },            
            
            'img_25438.jpg': {'rsp': ('c8', 'c0') },
            'img_67168.jpg': {'rsp': ('c8', 'c0') },

            # 'img_382.jpg'  : {'rsp': ('c9', 'c0') }, # Debatable    
            'img_16428.jpg': {'rsp': ('c9', 'c0') },            
            'img_60822.jpg': {'rsp': ('c9', 'c0') },
            'img_71334.jpg': {'rsp': ('c9', 'c0') },
            'img_79944.jpg': {'rsp': ('c9', 'c0') },                            
            'img_84986.jpg': {'rsp': ('c9', 'c0') },
            'img_89196.jpg': {'rsp': ('c9', 'c0') },
            'img_92682.jpg': {'rsp': ('c9', 'c0') },            
            'img_95888.jpg': {'rsp': ('c9', 'c0') }
            },
            
    'trnDups': [('img_70808.jpg', 'img_81828.jpg')]
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

glbImg = {
    'shape'         : (480, 640, 3), # Expected shape
    'size'          : 64,       # Pixel width and height.
    'pxlDepth'      : 255.0,    # Number of levels per pixel.
    'color'         : False,
#     'crop'          : {'x' : (-550, -70)},
#     'crop'          : {'x' : (0, 480)},    
    'crop'          : {'x' : (0 + 80, 480 + 80)},        
    'center_scale'  : True
    }

glbObsShuffleSeed = 127
glbTfVarSeed = 131

glbPickleFile = {
      'data' : 'data/img_D_SFDD_ImgSz_' + str(glbImg['size']) + '.pickle',
    'models' : 'data/img_M_SFDD_ImgSz_' + str(glbImg['size']) + '.pickle'
    }
        
print 'imported img_glbSpec_SFDD_Img_Sz_64.py'