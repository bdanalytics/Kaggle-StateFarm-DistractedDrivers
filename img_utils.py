def mydspVerboseTrigger(ix):
    return(
            ((ix >= 100000) and                   (ix % 200000 == 0)) or \
            ((ix >= 10000 ) and (ix < 100000) and (ix % 20000  == 0)) or \
            ((ix >= 1000  ) and (ix < 10000 ) and (ix % 2000   == 0)) or \
            ((ix >= 100   ) and (ix < 1000  ) and (ix % 200    == 0)) or \
            ((ix >= 10    ) and (ix < 100   ) and (ix % 20     == 0)) or \
            ((ix >= 1     ) and (ix < 10    ) and (ix % 2      == 0)) or \
             (ix ==  0    )
          )
          
# from nolearn.lasagne.visualize import plot_occlusion 
def occlusion_heatmap(net, x, target, square_length=7):
    """An occlusion test that checks an image for its critical parts.
    In this function, a square part of the image is occluded (i.e. set
    to 0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.
    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.
    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted by the
    batch iterator.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.
    target : int
      The true value of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).
    """
    
    import numpy as np    
    
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

#         num_classes = get_output_shape(net.layers_[-1])[1]
    num_classes = net.coef_.shape[0]
#         print 'occlusion_heatmap: num_classes: %d' % (num_classes)
    img = x[0].copy()
    bs, col, s0, s1 = x.shape

    heat_array = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((s1, col, s0, s1), dtype=img.dtype)
    probs = np.zeros((s0, s1, num_classes))

    # generate occluded images
    for i in range(s0):
        # batch s1 occluded images for faster prediction
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded[j] = x_pad[:, pad:-pad, pad:-pad]
#             y_proba = net.predict_proba(x_occluded)
        y_proba = net.predict_proba(np.reshape(x_occluded, (s1, s0 * s1)))            
        probs[i] = y_proba.reshape(s1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i in range(s0):
        for j in range(s1):
            heat_array[i, j] = probs[i, j, target]
    return heat_array

def _plot_heat_map(net, X, figsize, get_heat_image):
    import matplotlib.pyplot as plt
    
    if (X.ndim != 4):
        raise ValueError("This function requires the input data to be of "
                         "shape (b, c, x, y), instead got {}".format(X.shape))

    num_images = X.shape[0]
    if figsize[1] is None:
        figsize = (figsize[0], num_images * figsize[0] / 3)
    figs, axes = plt.subplots(num_images, 3, figsize=figsize)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

#         print '_plot_heat_map: X.shape: %s' % str(X.shape)
#         print '_plot_heat_map: range(num_images):'; print range(num_images)
    for n in range(num_images):
#             print '_plot_heat_map:  n: %d' % (n)
#             print '_plot_heat_map:  X:'; print X[n:n + 1, :, :, :]
        heat_img = get_heat_image(net, X[n:n + 1, :, :, :], n)

        ax = axes if num_images == 1 else axes[n]
        img = X[n, :, :, :].mean(0)
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest', cmap='Reds')
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest', cmap='Reds',
                     alpha=0.6)
        ax[2].set_title('super-imposed')
    return plt
            
def plot_occlusion(net, X, target, square_length=7, figsize=(9, None)):
    """Plot which parts of an image are particularly import for the
    net to classify the image correctly.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    X : numpy.array
      The input data, should be of shape (b, c, 0, 1). Only makes
      sense with image data.
    target : list or numpy.array of ints
      The true values of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at. If more than one sample is passed to X, each of them needs
      its own target.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    figsize : tuple (int, int)
      Size of the figure.
    Plots
    -----
    Figure with 3 subplots: the original image, the occlusion heatmap,
    and both images super-imposed.
    """
    return _plot_heat_map(
        net, X, figsize, lambda net, X, n: occlusion_heatmap(
            net, X, target[n], square_length))
    
def mydisplayImagePredictions(mdl, lclObsIdn, lclObsFtr, lclObsRsp, lclObsRspPredProba, 
                              lclRspClass, lclRspClassDesc):

    import matplotlib.pyplot as plt                              
    import numpy as np
    import os
    
    # Derive lclObsRsp from lclObsRspPredProba ???
                                  
#     print globals()

    for clsIx, cls in enumerate(lclRspClass):
        clsMsk = (np.argmax(lclObsRspPredProba, axis = 1) == clsIx)
        if not clsMsk.any(): continue
        
#         print 'mydisplayImagePredictions: %d' % (clsIx)
        clsObsRspPredProba = lclObsRspPredProba[clsMsk, :]
        clsObsIdn = [lclObsIdn[ixMsk] for ixMsk in xrange(len(lclObsIdn)) \
                        if clsMsk[ixMsk]]
        print '\n'

        maxClsProba = np.max(clsObsRspPredProba[:, clsIx])
        maxObsRspPredProba = clsObsRspPredProba[:, clsIx] == maxClsProba
        print 'Max Proba for cls: %s; desc: %s; proba: %0.4f; nObs: %d' % \
            (cls, lclRspClassDesc[cls], maxClsProba, maxObsRspPredProba.sum())
        idnIx = np.argmax(clsObsRspPredProba[:, clsIx])    
        print '  %s:' % clsObsIdn[idnIx]
        
#         imgFilePth = os.getcwd() + '/data/' + glbDataFile['newFoldersPth'] + '/' + \
#                         clsObsIdn[np.argmax(clsObsRspPredProba[:, clsIx])]
#         print '  %s:' % imgFilePth
#         jpgfile = Image(imgFilePth, format = 'jpg', 
#                             width = glbImg['size'] * 4, height = glbImg['size'] * 4)
#         display(jpgfile)

        plot_occlusion(mdl, np.reshape(lclObsFtr[idnIx], 
            (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
                       lclObsRsp[idnIx:(idnIx + 1)])
        plt.show()         
        print '  Proba:'; 
        print np.array_str(clsObsRspPredProba[np.argmax(clsObsRspPredProba[:, clsIx]), :],
                           precision=4, suppress_small=True)

        minClsProba = np.min(clsObsRspPredProba[:, clsIx])
        minObsRspPredProba = clsObsRspPredProba[:, clsIx] == minClsProba
        print 'Min Proba for cls: %s; desc: %s; proba: %0.4f; nObs: %d' % \
            (cls, lclRspClassDesc[cls], minClsProba, minObsRspPredProba.sum())
        idnIx = np.argmin(clsObsRspPredProba[:, clsIx])    
        print '  %s:' % clsObsIdn[idnIx]
        
#         imgFilePth = os.getcwd() + '/data/' + glbDataFile['newFoldersPth'] + '/' + \
#                         clsObsIdn[np.argmin(clsObsRspPredProba[:, clsIx])]
#         print '  %s:' % imgFilePth
#         jpgfile = Image(imgFilePth, format = 'jpg', 
#                             width = glbImg['size'] * 4, height = glbImg['size'] * 4)
#         display(jpgfile)

        plot_occlusion(mdl, np.reshape(lclObsFtr[idnIx], 
            (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
                       lclObsRsp[idnIx:(idnIx + 1)])
        plt.show()         
        print '  Proba:'; 
        print np.array_str(clsObsRspPredProba[np.argmin(clsObsRspPredProba[:, clsIx]), :],
                           precision=4, suppress_small=True)
        thsObsRspPredProba = clsObsRspPredProba[np.argmin(clsObsRspPredProba[:, clsIx]), :]
        thsObsRspPredProba[clsIx] = 0
        print '  next best class: %s' % \
            (lclRspClassDesc[lclRspClass[np.argmax(thsObsRspPredProba)]])          
          
def myexpandGrid(dct):
    from itertools import product
    import pandas as pd
        
    return pd.DataFrame([row for row in product(*dct.values())], 
                       columns=dct.keys())
                       
def myimportDbs(filePathName):
    from six.moves import cPickle as pickle

    global glbObsFitIdn
    global glbObsFitFtr
    global glbObsFitRsp
            
    global glbObsVldIdn, glbObsVldFtr, glbObsVldRsp     
    global glbObsNewIdn, glbObsNewFtr, glbObsNewRsp
    global sbtNewCorDf         

    with open(filePathName, 'rb') as f:
      print 'Importing database from %s...' % (filePathName)  
      save = pickle.load(f)

    #   glbObsTrnIdn = save['glbObsTrnIdn']
    #   glbObsTrnFtr = save['glbObsTrnFtr']
    #   glbObsTrnRsp = save['glbObsTrnRsp']
    
      glbObsFitIdn = save['glbObsFitIdn']
      glbObsFitFtr = save['glbObsFitFtr']
      glbObsFitRsp = save['glbObsFitRsp']

      glbObsVldIdn = save['glbObsVldIdn']
      glbObsVldFtr = save['glbObsVldFtr']
      glbObsVldRsp = save['glbObsVldRsp']

      glbObsNewIdn = save['glbObsNewIdn']
      glbObsNewFtr = save['glbObsNewFtr']
      glbObsNewRsp = save['glbObsNewRsp']

      sbtNewCorDf = save['sbtNewCorDf']

      del save  # hint to help gc free up memory 
      
#       print '   returning from myimportDbs'
#       print globals().keys().index('glbObsFitIdn')
#       print len(glbObsFitIdn)
      
#       return                      
      return  glbObsFitIdn, glbObsFitFtr, glbObsFitRsp, \
              glbObsVldIdn, glbObsVldFtr, glbObsVldRsp, \
              glbObsNewIdn, glbObsNewFtr, glbObsNewRsp, \
              sbtNewCorDf, \
              None

def myexportDf(retResultsDf, save_filepathname, save_drop_cols = None):
    import os

    if not (save_drop_cols == None):                       
        try:
            tmpResultsDf = retResultsDf.drop(save_drop_cols, axis = 1)
        except ValueError, e:
            print(e)
            tmpResultsDf = retResultsDf
        except Exception, e:
            print(e)    
            raise
    else: tmpResultsDf = retResultsDf        

    try:
        tmpResultsDf.to_pickle(save_filepathname)
    except Exception, e:
        print(e)    
        raise

    print 'Compressed pickle file: %s; size: %d KB'% (save_filepathname, 
                            os.stat(save_filepathname).st_size / 1024)
    return None                            
                       
                       
def mysearchParams(thsFtn, srchParamsDct = {}, curResultsDf = None, mode = 'displayonly', 
                    sort_values = None, sort_ascending = None,
                    save_drop_cols = None, save_filepathname = None,
                    **kwargs):
    import os                    
    import pandas as pd
    from six.moves import cPickle as pickle    
    
#     print '  kwargs: %s' % (kwargs)

    if not isinstance(curResultsDf, pd.DataFrame) and \
       (curResultsDf == None) and \
       not (save_filepathname == None):
        print 'mysearchParams: importing curResultsDf from %s...' % (save_filepathname)
#         print 'mysearchParams: os.listdir():'; print os.listdir('data/')
#         print 'wtf'
#         return None
        try:
            with open(save_filepathname, 'rb') as f:
                curResultsDf = pickle.load(f)
                assert isinstance(curResultsDf, pd.DataFrame), 'type(curResultsDf): %s, expecting pd.DataFrame' % (str(type(curResultsDf)))            
        except IOError, e:
            print 'mysearchParams: %s; assigning pd.DataFrame() to curResultsDf' % (e)
            curResultsDf = pd.DataFrame()

    retResultsDf = curResultsDf
    
    srchParamsDf = myexpandGrid(srchParamsDct).set_index(srchParamsDct.keys(), drop = False)
    
    try:
        chkResultsDf = retResultsDf[srchParamsDct.keys()].set_index(srchParamsDct.keys(), drop = False)
    except KeyError, e:
        print 'mysearchParams: %s of curResultsDf' % (e)
        chkResultsDf = retResultsDf
    except TypeError, e:
        print 'mysearchParams: curResultsDf: %s' % (e)
        chkResultsDf = None
    
    try:
        runResultsDf = chkResultsDf.join(srchParamsDf, on = srchParamsDct.keys(), how = 'right',
                                lsuffix = '.avl', rsuffix = '.srch')
    except KeyError, e:
        print 'mysearchParams: %s not in curResultsDf' % (e)    
        runResultsDf = srchParamsDf
        runResultsDf[str(e) + '.right'] = None
    
#     print '  before filter runResultsDf:'; print(runResultsDf)
    # Filter results that already exist
    runResultsDf = runResultsDf[pd.isnull(runResultsDf).apply(any, axis = 1)][srchParamsDct.keys()]
        
    if (mode == 'displayonly'):
        print 'mysearchParams: will run %s with params:' % (thsFtn)
        print runResultsDf
    else:        
        for rowIx in xrange(runResultsDf.shape[0]):
            srchKwargs = kwargs.copy() 
            srchKwargs.update(runResultsDf.iloc[rowIx].to_dict())
            
            print 'mysearchParams: running %s with params:' % (thsFtn)
            print runResultsDf.iloc[rowIx] 

            # Function expects first return value to be a pd.DataFrame                 
            thsResults = thsFtn(**srchKwargs)
#             print 'mysearchParams: thsResults:'; print thsResults            
            if isinstance(thsResults, tuple):
                assert isinstance(thsResults[0], pd.DataFrame), \
                    '%s returns first object whose type is %s, expecting pd.DataFrame' % \
                        (thsFtn, str(type(thsResults[0])))
                thsResultsDf = thsResults[0]                        
#                 print 'mysearchParams: thsResultsDf:'; print thsResultsDf                
            else:                        
                assert isinstance(thsResults, pd.DataFrame), \
                    '%s returns object whose type is %s, expecting pd.DataFrame' % \
                        (thsFtn, str(type(thsResults)))
                thsResultsDf = thsResults                        
            
            retResultsDf = retResultsDf.append(thsResultsDf)  
            
#     retResultsDf.ix[retResultsDf['bstFit'].isnull(), 'bstFit'] = False

    # Set up dataframe for printing index which is useful in scanning key rows
#     print 'mysearchParams: retResultsDf:'; print retResultsDf
    if (retResultsDf.shape[0] > 0):
        retResultsDf = retResultsDf.set_index(srchParamsDct.keys(), drop = False)
        if   not (sort_values == None) and not (sort_ascending == None):
    #         print '  sort_values and sort_ascending'
            retResultsDf = retResultsDf.sort_values(sort_values, ascending = sort_ascending)
        elif not (sort_values == None):   
            retResultsDf = retResultsDf.sort_values(sort_values)    
    #     retResultsDf = (retResultsDf
    #                 .set_index(srchParamsDct.keys(), drop = False)
    # #                 .sort_values(['logLossVld', 'accVld'], ascending = 
    # #                              ['True'      , 'False' ])
    #                 )

    if not (mode == 'displayonly'):
        print(retResultsDf[list(set(retResultsDf.columns) - set(srchParamsDct.keys()))])
    
    # Save retResultsDf
    if not (save_filepathname == None):
        myexportDf(retResultsDf, save_filepathname, save_drop_cols)
                    
    return(retResultsDf)
    
def mywriteSubmission(lclObsNewRspPredProba, fileName):
    import pandas as pd

    sbmObsNewDf = pd.DataFrame(lclObsNewRspPredProba)
    sbmObsNewDf.columns = glbRspClass
    sbmObsNewDf['img'] = glbObsNewIdn
    sbmObsNewDf = (sbmObsNewDf
                    .set_index(['img'], 
                               drop = False)
                    .sort_values('img')
                    )
    sbmObsNewDf = sbmObsNewDf[['img'] + glbRspClass]
    print sbmObsNewDf.head()
    print sbmObsNewDf.tail()
    
    print '\nexporting %d rows to %s...' % (sbmObsNewDf.shape[0], 
                                         fileName)
    sbmObsNewDf.to_csv(fileName, index = False)            