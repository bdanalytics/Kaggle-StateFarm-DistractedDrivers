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
def occlusion_heatmap(net, x, target, square_length=7, 
                        tfwXOcc = None, tfwYOccPby = None):
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
    import tensorflow as tf    
    
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

#         num_classes = get_output_shape(net.layers_[-1])[1]
    if not (getattr(net, 'coef_', None) == None): num_classes = net.coef_.shape[0]
    elif isinstance(net, tf.Graph): 
#         print(dir(net))
        # Assumes last trainable_variable.shape contains num_classes
        num_classes = net.get_collection_ref('trainable_variables')[-1].get_shape().as_list()[0]
    elif isinstance(net, tf.Session): 
#         print(dir(net))
        # Assumes last trainable_variable.shape contains num_classes
        num_classes = net.graph.get_collection_ref('trainable_variables')[-1].get_shape().as_list()[0]
    else: 
        print 'occlusion_heatmap: unknown object type: %s' % str(type(net)) 
        print(dir(net))
        raise TypeError
            
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
#         print 'occlusion_heatmap: x_occluded.shape: %s' % (str(x_occluded.shape))
        if not isinstance(net, tf.Session):
            y_proba = net.predict_proba(np.reshape(x_occluded, (s1, s0 * s1)))
        else:                
            y_proba = net.run(tfwYOccPby, feed_dict = {tfwXOcc: np.reshape(x_occluded, (s1, s0 * s1))})
#         print 'occlusion_heatmap: y_proba.shape: %s' % (str(y_proba.shape))            
#             sess.run(accuracy, feed_dict={x: test_dataset.images, y_: test_dataset.labels})            
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
#     print "_plot_heat_map: figsize: %s" % str(figsize)
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
            
def plot_occlusion(net, X, target, square_length=7, figsize=(9, None), 
                    tfwXOcc = None, tfwYOccPby = None):
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
            net, X, target[n], square_length, tfwXOcc, tfwYOccPby))
    
# https://github.com/Elucidation/tensorflow_chessbot/blob/master/helper_functions.py
# def display_weight(a, fmt='jpeg', rng=[0,1]):
def display_weight(X, a, fmt='jpeg', rng=[0,1]):
    """Display an array as a color picture."""

    import cStringIO
    from IPython.display import Image, display  
    import numpy as np
    import PIL  

    a = (a - rng[0])/float(rng[1] - rng[0]) # normalized float value
    a = np.uint8(np.clip(a*255, 0, 255))
    f = cStringIO.StringIO()

    v = np.asarray(a, dtype=np.uint8)

    # blue is high intensity, red is low
    # Negative
    r = 255-v.copy()
    r[r<127] = 0
    r[r>=127] = 255
#     print 'display_weight: # of pixels with r == 255: %d' % np.sum(r == 255)

    # None
    g = np.zeros_like(v)

    # Positive
    b = v.copy()
    b[b<127] = 0
    b[b>=127] = 255
#     print 'display_weight: # of pixels with b == 255: %d' % np.sum(b == 255)    

    #np.clip((v-127)/2,0,127)*2

    #-1 to 1
    intensity = np.abs(2.*a-1)

    rgb = np.uint8(np.dstack([r,g,b]*intensity))
#     print 'display_weight: rgb.shape: %s' % str(rgb.shape)

    PIL.Image.fromarray(rgb).save(f, fmt)
    display(Image(data=f.getvalue(), width=100))
  
    import matplotlib.pyplot as plt
    
#     if (X.ndim != 4):
#         raise ValueError("This function requires the input data to be of "
#                          "shape (b, c, x, y), instead got {}".format(X.shape))

    num_images = 1; figsize = (9, None)
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
#         heat_img = get_heat_image(net, X[n:n + 1, :, :, :], n)

        ax = axes if num_images == 1 else axes[n]
#         img = X[n, :, :, :].mean(0)
#         print 'display_weight: X[n].shape: %s' % str(X[n].shape)
        img = X[n].reshape((X[n].shape[1], X[n].shape[2]))
#         print 'display_weight: img.shape: %s' % str(img.shape)
        heat_img = rgb
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest')
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest',
                     alpha=0.6)
        ax[2].set_title('super-imposed')
        
    return plt
                                          
def mydisplayImagePredictions(mdl, W, lclObsIdn, lclObsFtr, lclObsRsp, lclObsRspPredProba, 
                              lclRspClass, lclRspClassDesc, imgVisualFn = None, 
                              tfwXOcc = None, tfwYOccPby = None):

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

        for typPby in ['max', 'min']:
            typClsPby = np.max(clsObsRspPredProba[:, clsIx]) if typPby == 'max' else \
                        np.min(clsObsRspPredProba[:, clsIx])
            typClsYPby = clsObsRspPredProba[:, clsIx] == typClsPby
            print '%s Pby for cls: %s; desc: %s; proba: %0.4f; nObs: %d' % \
                (typPby, cls, lclRspClassDesc[cls], typClsPby, typClsYPby.sum())
            idnIx = np.argmax(clsObsRspPredProba[:, clsIx])  if typPby == 'max' else \
                    np.argmin(clsObsRspPredProba[:, clsIx])              
            print '  %s:' % clsObsIdn[idnIx]
        
    #         imgFilePth = os.getcwd() + '/data/' + glbDataFile['newFoldersPth'] + '/' + \
    #                         clsObsIdn[np.argmax(clsObsRspPredProba[:, clsIx])]
    #         print '  %s:' % imgFilePth
    #         jpgfile = Image(imgFilePth, format = 'jpg', 
    #                             width = glbImg['size'] * 4, height = glbImg['size'] * 4)
    #         display(jpgfile)

    #         assert imgVisualFn == display_weight, 'imgVisualFn not recognized as display_weight'
#             if (imgVisualFn == plot_occlusion):
#                 imgVisualFn(mdl, np.reshape(lclObsFtr[idnIx], 
#                     (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
#                                lclObsRsp[idnIx:(idnIx + 1)], 
#                                tfwXOcc = tfwXOcc, tfwYOccPby = tfwYOccPby)
#             elif (imgVisualFn == display_weight):
#     #             print 'mydisplayImagePredictions: W.shape: %s' % str(W.shape)
#                 imgSz = int(np.sqrt(W.shape[0]))
#                 imgVisualFn(np.reshape(lclObsFtr[idnIx], 
#                     (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
#                     np.reshape(W[:, clsIx], (imgSz, imgSz)))
#             else:
#                 raise ValueError('unsupported imgVisualFn: %s' % imgVisualFn)
#           
                          
            if (imgVisualFn in [None, plot_occlusion]):
                print "  plot_occlusion:"                         
                plot_occlusion(mdl, np.reshape(lclObsFtr[idnIx], 
                        (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
                                   lclObsRsp[idnIx:(idnIx + 1)], 
                                   tfwXOcc = tfwXOcc, tfwYOccPby = tfwYOccPby)
                plt.show()         

            if (imgVisualFn in [None, display_weight]):
                print "  display_weight:"
                imgSz = int(np.sqrt(W.shape[0]))
                display_weight(np.reshape(lclObsFtr[idnIx], 
                    (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
                    np.reshape(W[:, clsIx], (imgSz, imgSz)))                                   
                plt.show()         
            
            print '  Proba:'; 
            print np.array_str(clsObsRspPredProba[idnIx, :],
                               precision=4, suppress_small=True)
            if typPby == 'min':                   
                thsObsRspPredProba = clsObsRspPredProba[idnIx, :]
                thsObsRspPredProba[clsIx] = 0
                print '  next best class: %s' % \
                    (lclRspClassDesc[lclRspClass[np.argmax(thsObsRspPredProba)]])          
                               

#         minClsProba = np.min(clsObsRspPredProba[:, clsIx])
#         minObsRspPredProba = clsObsRspPredProba[:, clsIx] == minClsProba
#         print 'Min Proba for cls: %s; desc: %s; proba: %0.4f; nObs: %d' % \
#             (cls, lclRspClassDesc[cls], minClsProba, minObsRspPredProba.sum())
#         idnIx = np.argmin(clsObsRspPredProba[:, clsIx])    
#         print '  %s:' % clsObsIdn[idnIx]
#         
# #         imgFilePth = os.getcwd() + '/data/' + glbDataFile['newFoldersPth'] + '/' + \
# #                         clsObsIdn[np.argmin(clsObsRspPredProba[:, clsIx])]
# #         print '  %s:' % imgFilePth
# #         jpgfile = Image(imgFilePth, format = 'jpg', 
# #                             width = glbImg['size'] * 4, height = glbImg['size'] * 4)
# #         display(jpgfile)
# 
#         if (imgVisualFn == plot_occlusion):
#             imgVisualFn(mdl, np.reshape(lclObsFtr[idnIx], 
#                 (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
#                            lclObsRsp[idnIx:(idnIx + 1)])
#         elif (imgVisualFn == display_weight):
# #             print 'mydisplayImagePredictions: W.shape: %s' % str(W.shape)
#             imgSz = int(np.sqrt(W.shape[0]))
#             imgVisualFn(np.reshape(lclObsFtr[idnIx], 
#                 (1, 1, lclObsFtr.shape[1], lclObsFtr.shape[2])), 
#                 np.reshape(W[:, clsIx], (imgSz, imgSz)))
#         else:
#             raise ValueError('unsupported imgVisualFn: %s' % imgVisualFn)
#                                    
#         plt.show()         
#         print '  Proba:'; 
#         print np.array_str(clsObsRspPredProba[np.argmin(clsObsRspPredProba[:, clsIx]), :],
#                            precision=4, suppress_small=True)
#         thsObsRspPredProba = clsObsRspPredProba[np.argmin(clsObsRspPredProba[:, clsIx]), :]
#         thsObsRspPredProba[clsIx] = 0
#         print '  next best class: %s' % \
#             (lclRspClassDesc[lclRspClass[np.argmax(thsObsRspPredProba)]])   
    return None       
          
def myexpandGrid(dct):
    from itertools import product
    import pandas as pd
        
    return pd.DataFrame([row for row in product(*dct.values())], 
                       columns=dct.keys())
                       
# To ensure Kaggle evaluation metric is same as sklearn.metrics.log_loss
def mygetMetricLogLoss(lclYHen, lclYPby, returnTyp = 'total', verbose = False):

    from collections import namedtuple
    import numpy as np

    assert lclYHen.ndim == 2, \
        "mygetMetricLogLoss: expecting lclYHen as hot-encoded np.array with ndim = 2 vs. %d" % \
            (lclYHen.ndim)
            
    spdReturnTyp = ['total', 'class', 'outlier']        
    assert returnTyp in spdReturnTyp, "unsupported returnTyp: %s; supported: %s" % \
        (returnTyp, spdReturnTyp)

    lclY = np.argmax(lclYHen, axis = 1)

    lclYIndicator = np.zeros_like(lclYPby)
    for cls in xrange(lclYIndicator.shape[1]):
        lclYIndicator[lclY == cls, cls] = 1

    # Scale proba to sum to 1 for each row
    tmpYPby = lclYPby
    sclYPbyRowSum = tmpYPby.sum(axis = 1)
    sclYPbyRowSumChk = (np.abs(sclYPbyRowSum - 1.0) > 1e-15)
    if (sclYPbyRowSumChk.sum() > 0):
        maxDff = np.max(np.abs(sclYPbyRowSum - 1.0))
        if maxDff > 1e-06:
            print 'mygetMetricLogLoss: row sums != 1 for %d (of %d) obs; max diff: %.4e' % \
                (sclYPbyRowSumChk.sum(), sclYPbyRowSumChk.shape[0], maxDff)
#         print sclYPbyRowSum[sclYPbyRowSumChk]
#         print(np.vectorize("%.4e".__mod__)(sclYPbyRowSum[sclYPbyRowSumChk][:5] - 1.0))
#         print "mygetMetricLogLoss: tmpYPby.shape: %s" % (str(tmpYPby.shape))
#         print "mygetMetricLogLoss: sclYPbyRowSum.shape: %s" % (str(sclYPbyRowSum.shape))
        sclYPbyRow = np.ones(tmpYPby.shape)
        for rowIx in xrange(sclYPbyRow.shape[0]):
            sclYPbyRow[rowIx, :] = sclYPbyRowSum[rowIx]                            
        sclYPby = tmpYPby / sclYPbyRow
#         print "mygetMetricLogLoss: sclYPby.shape: %s" % (str(sclYPby.shape))        
        tmpYPby = sclYPby
        
    # Bound proba to limit log fn outliers
    bndYPby = tmpYPby
    bndYPby[bndYPby > 1-1e-15] = 1-1e-15
    bndYPby[bndYPby < 0+1e-15] = 0+1e-15
    nModProba = (tmpYPby != bndYPby).sum()
    if (nModProba > 0):
        print 'mygetMetricLogLoss: minmax of probabilities modified %d cells' % (nModProba)
    tmpYPby = bndYPby    
    
    logLossObs = (lclYIndicator * np.log(tmpYPby)).sum(axis = 1)
    logLossCls = (lclYIndicator * np.log(tmpYPby)).sum(axis = 0) / \
                  tmpYPby.shape[0]
#                   np.unique(lclY, return_counts = True)[1]                      
    if verbose:
        print 'mygetMetricLogLoss: logLossObs outlier: %.4f; ix: %d' % \
            (-np.min(logLossObs), np.argmin(logLossObs))
        print "mygetMetricLogLoss: logLossCls:"; print -logLossCls;
        print "mygetMetricLogLoss: logLossClsSum: %0.4f" % (-logLossCls.sum())
#         print "mygetMetricLogLoss: logLossClsSum: %0.4f" % (-logLossCls.sum() / tmpYPby.shape[1])        
    logLoss = 0 - (logLossObs.sum() / tmpYPby.shape[0])
    
    Outlier = namedtuple('Outlier', 'ix, logLoss')
    maxOutlier = Outlier(np.argmin(logLossObs), -np.min(logLossObs))
    returnVal = {   
                    'total'     : logLoss, 
                    'class'     : -logLossCls,
                    'outlier'   : maxOutlier
                }
    
    return(returnVal.get(returnTyp, 'total'))
                                              
def myimportDbs(filePathName):
    from six.moves import cPickle as pickle

    # Tried globals outside fn, inside fn & with f; nothing works

    with open(filePathName, 'rb') as f:
        print 'Importing database from %s...' % (filePathName)  
        save = pickle.load(f)
        
#         global glbObsFitIdn, glbObsFitFtr, glbObsFitRsp
#         global glbObsVldIdn, glbObsVldFtr, glbObsVldRsp     
#         global glbObsNewIdn, glbObsNewFtr, glbObsNewRsp
#         global sbtNewCorDf         

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

    #     del save  # hint to help gc free up memory 
      
        return  glbObsFitIdn, glbObsFitFtr, glbObsFitRsp, \
                glbObsVldIdn, glbObsVldFtr, glbObsVldRsp, \
                glbObsNewIdn, glbObsNewFtr, glbObsNewRsp, \
                sbtNewCorDf, \
                None # Dummy Placeholder
#     return None

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
        print runResultsDf.to_string(index = False)
        print 'mysearchParams: total runs: %d' % (runResultsDf.shape[0])
    else:        
        if runResultsDf.shape[0] > 5:
            print "mysearchParams: number of runs: %2d; running first 5 only" % (runResultsDf.shape[0])
            runResultsDf = runResultsDf.iloc[:5]
        for rowIx in xrange(runResultsDf.shape[0]):
            srchKwargs = kwargs.copy() 
            srchKwargs.update(runResultsDf.iloc[rowIx].to_dict())
            
            print 'mysearchParams: running %s with params:' % (thsFtn)
#             print runResultsDf.iloc[rowIx].to_string(index = False)
            print runResultsDf.iloc[rowIx].to_string(index = True)             

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
        missKey = list(set(srchParamsDct.keys()).difference(set(retResultsDf.columns)))
        for key in missKey:
            retResultsDf[key] = None
        
        retResultsDf = retResultsDf.set_index(['id'] + srchParamsDct.keys(), drop = False)
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
        print(retResultsDf[list(set(retResultsDf.columns) - set(['id'] + srchParamsDct.keys()))])
    
    # Save retResultsDf
    if (save_filepathname != None) and (mode != 'displayonly'):
        myexportDf(retResultsDf, save_filepathname, save_drop_cols)
                    
    return(retResultsDf)                