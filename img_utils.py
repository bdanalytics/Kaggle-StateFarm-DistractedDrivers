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
          
def myexpandGrid(dct):
    from itertools import product
    import pandas as pd
        
    return pd.DataFrame([row for row in product(*dct.values())], 
                       columns=dct.keys())
                       
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
        print '  importing curResultsDf from %s...' % (save_filepathname)
        with open(save_filepathname, 'rb') as f:
#             tmpDct = pickle.load(f)
#             assert (len(tmpDct.keys()) == 1), \
#                 'too many objects in pickled file: %s' % (tmpDct.keys())
#             curResultsDf = tmpDct.values[0]
            curResultsDf = pickle.load(f)
            assert isinstance(curResultsDf, pd.DataFrame), 'type(curResultsDf): %s, expecting pd.DataFrame' % (str(type(curResultsDf)))            
#             del tmpDct  # hint to help gc free up memory

    retResultsDf = curResultsDf
    
    srchParamsDf = myexpandGrid(srchParamsDct).set_index(srchParamsDct.keys(), drop = False)
    
    try:
        chkResultsDf = retResultsDf[srchParamsDct.keys()].set_index(srchParamsDct.keys(), drop = False)
    except KeyError, e:
        print '%s of curResultsDf' % (e)
        chkResultsDf = retResultsDf
    except TypeError, e:
        print 'curResultsDf: %s' % (e)
        chkResultsDf = None
    
    try:
        runResultsDf = chkResultsDf.join(srchParamsDf, on = srchParamsDct.keys(), how = 'right',
                                lsuffix = '.avl', rsuffix = '.srch')
    except KeyError, e:
        print '%s not in curResultsDf' % (e)    
        runResultsDf = srchParamsDf
        runResultsDf[str(e) + '.right'] = None
    
#     print '  before filter runResultsDf:'; print(runResultsDf)
    # Filter results that already exist
    runResultsDf = runResultsDf[pd.isnull(runResultsDf).apply(any, axis = 1)][srchParamsDct.keys()]
        
    if (mode == 'displayonly'):
        print 'Running %s with params:' % (thsFtn)
        print runResultsDf
    else:        
        for rowIx in xrange(runResultsDf.shape[0]):
            srchKwargs = kwargs.copy() 
            srchKwargs.update(runResultsDf.iloc[rowIx].to_dict())

            # Function expects first return value to be a pd.DataFrame                 
            thsResults = thsFtn(**srchKwargs)
            if isinstance(thsResults, tuple):
                assert isinstance(thsResults[0], pd.DataFrame), \
                    '%s returns first object whose type is %s, expecting pd.DataFrame' % \
                        (thsFtn, str(type(thsResults[0])))
                thsResultsDf = thsResults[0]                        
            else:                        
                assert isinstance(thsResults, pd.DataFrame), \
                    '%s returns object whose type is %s, expecting pd.DataFrame' % \
                        (thsFtn, str(type(thsResults)))
                thsResultsDf = thsResults                        
            
            retResultsDf = retResultsDf.append(thsResultsDf)  
            
#     retResultsDf.ix[retResultsDf['bstFit'].isnull(), 'bstFit'] = False

    # Set up dataframe for printing index which is useful in scanning key rows
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
#     import os
    
    if not (save_filepathname == None):
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
                    
#     if (mode == 'displayonly'):
#         return(None)
#     else:
#         return(retResultsDf)
    return(retResultsDf)        