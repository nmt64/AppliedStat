#Final Project Data Mining
#Author: Duc Nguyen - Minh Vu
#Professor: Steve Bogaerts


####NOTE: We commented a lot of our codes out (green) due to the fact 
#           that some of them are meant to be run on XSEDE and the other 
#           meant to be run on our personal laptops.

#         For example, a lot of the parameters tuning processes are expensive so I need to run them on XSEDE
#         So when I run something on my personal laptop, I need to comment them out so that 
#        my program can run faster, and the writefile() function can only be run on my laptop, not XSEDE

import random
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
#from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.cross_validation import KFold
import sklearn.model_selection
from sklearn.linear_model import Ridge
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm



#read Data function
def readData():
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")
    
    return trainDF,testDF

#write result file function
def writefile(predictions,testDF,name):
    #create csv file for submission
    submitDF = pd.DataFrame(
            {'Id': testDF.loc[:,'Id'],
             'SalePrice': predictions }
            )
    submitDF.to_csv(name, index=False)
    
#standardizing function for manipulating the data 
def standardize(df, listCol):
    mean = df.loc[:,listCol].mean()
    deviation = df.loc[:,listCol].std()
    df.loc[:,listCol] = df.loc[:,listCol] - mean
    df.loc[:,listCol] = df.loc[:,listCol]/deviation
    
#normalizing function for manipulating the data  
def normalize(df,someCols):
    top = df.loc[:,someCols] - df.loc[:,someCols].min()
    bot = df.loc[:,someCols].max() - df.loc[:,someCols]
    df.loc[:,someCols] = top/bot
    
    
#basic preprocessing function
def preprocessing(df,sourceDF):
    
    #feature engineering : adding a new variable 'YardSize; by using LotArea and GrLivArea
        #trainDF.loc[:,'YrOld'] = trainDF.loc[:,'YrSold']- trainDF.loc[:,'YearBuilt']
        #print(trainDF)
    df.loc[:,'YardSize'] = df.loc[:,'LotArea']- df.loc[:,'GrLivArea']
    sourceDF.loc[:,'YardSize'] = sourceDF.loc[:,'LotArea']- sourceDF.loc[:,'GrLivArea']
    
    #the correlation matrix using .corr function
    corrmatrix = sourceDF.corr()
    
    #then using the heat map to see the correlation between attributes
    #f, ax = plt.subplots(figsize=(8, 6))
    #sns.heatmap(corrmatrix, vmax=.8, square=True);
    
    
   
    # BEGIN: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: create saleprice correlation matrix heatmap to see the 10 attributes least correlated to the output column('SalePrice')

    k = 10 #number of variables for heatmap or to find the 10 least correlated attributes to SalePrice
    cols = corrmatrix.nsmallest(k, 'SalePrice')['SalePrice'].index
    #cm = np.corrcoef(sourceDF[cols].values.T)
    #sns.set(font_scale=1.25)
    #hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    #plt.show()
    
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    
    #print out correlation list to have a better understanding of the correlation values
    corr_list = corrmatrix['SalePrice'].sort_values(axis=0,ascending=False).iloc[1:]
    print(corr_list)
    #return cols
    #drop the 10 least correlated attributes
    df = df.drop(cols,axis=1)

    #return len(trainDF.columns)
    
    #Dealing with missing values
    
    #creating a sorted series of the attributes with most to fewer missing values.
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    
    #print out the sorted series
    #missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    #print(missing_data.head(20))
    
    #drop columns with >50% missing values
    misCol = percent[percent>0.50].index
    df = df.drop(misCol,axis=1)

    #print(trainDF.columns)
    #print(len(trainDF.columns))
    
    #spliting the attributes into number attributes and categorical attributes.
    catcolumns = [col for col in df.columns.values if df[col].dtype == 'object']
    data_cat = df[catcolumns]
    data_num = df.drop(catcolumns, axis=1)
    
    #print('data Number',data_num.head(1))
    #print('data Category',data_cat.head(1))
    
    #Fill NaN for both types of attributes: 
        #mean for number attributes
        #mode for categorical attributes
    for col in data_num.columns.values:
        df[col] =df[col].fillna(df[col].mean())
    for col in data_cat.columns.values:
        df[col] =df[col].fillna(df[col].mode())
    
    #print(trainDF)
    
    #drop SalePrice column if preprocessing the training data set.
    if 'SalePrice' in df.columns:
        df=df.drop('SalePrice',axis=1)
    
    return df

#A way to try to drop the outliers from the data set which I don't use because 
def outliers1():
    
    train, test = readData()
    
    test1 = pd.read_csv("submission1.csv").drop('Id',axis=1,inplace=False)
    origin = pd.DataFrame(train['SalePrice'])
    
    #dropping all result rows that has a difference with the original SalePrice of more than 20000
    dif = np.abs(test1-origin) > 20000
    idx = dif[dif['SalePrice']].index.tolist()
    train.drop(train.index[idx],inplace=True)
    print(train.shape)
    print(len(idx))
    
#Another way that I used to remove outliers based on data visualization
def outliers2(df_train):

    var = 'GrLivArea'
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    df_train = df_train.drop(df_train.loc[:,"GrLivArea"].nlargest(2).index)
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    
    
    var = 'GarageArea'
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    df_train = df_train.drop(df_train.loc[:,var].nlargest(4).index)
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#    
    var = 'TotalBsmtSF'
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
    df_train = df_train.drop(df_train.loc[:,"TotalBsmtSF"].nlargest(1).index)
   
    return df_train


#Manual Gradient Descent function to return theta array
    
def gradientDescent(train,output):

    theta = np.empty(train.shape[1],dtype=float)
    for i in range(train.shape[1]):
        theta[i] = random.uniform(-0.5,0.5)
   # print(theta)
    '''   
    h_x = np.empty(len(theta,dtype=float))
    for j in range(len(h_x)):
        for k in range(len(theta)):
            h_x[j]=  theta[k]*'''
    df = train.copy()
    
    alpha =0.01
    '''
    for k in range(1,2):
        #df.loc[:,"h_x"] = theta[0]*1 + theta[1]*df.loc[0,0] + theta[2]*df.loc[0,1]
        df.loc[:,"h_x"] = df.apply(lambda x: a(x, theta),axis=1)
    
        num=train.shape[0]
       # print(num)
       # print(len(theta))
       '''
    for k in range(1,500):
            #df.loc[:,"h_x"] = theta[0]*1 + theta[1]*df.loc[0,0] + theta[2]*df.loc[0,1]
            df.loc[:,"h_x"] = df.apply(lambda x: a(x, theta),axis=1)
            #print(df.loc[:,'h_x'])
            num=df.shape[0]
            #print(num)
            #print(list(range(len(theta))))
            #print(df)
            
            ax1 = (df.loc[:,'h_x'] - output).sum()
            theta[0]= theta[0] - ax1*alpha/num
    
            for i in range(1,len(theta)):
                ax = df.loc[:,"h_x"]*df.iloc[:,i-1]
                bx = output*df.iloc[:,i-1]
                sum1 = (ax-bx).sum()
                theta[i] = theta[i]- sum1*alpha/num
        #print(theta)
            print(k)
    return theta



#helper function for gradient descent
def a(testrow,theta):
    sum1=theta[0]
    for i in range(1,len(theta)):
        sum1 = sum1 +theta[i]*testrow[i-1]
        
    return sum1      


#TESTING GRADIENT DESCENT FUNCTION
def testGradDescent():
    datatest= {'col1': [0.026, -0.513, 0.487], 'col2': [-0.33, -0.33, 0.67], 'col3': [0.098,-0.549,0.451]}
    dtest = pd.DataFrame(data=datatest)
    
    #print(dtest)
    theta= [0.5,-0.2,0.3]
   # print(theta)
    alpha = 0.02
    output=(dtest.loc[:,'col3'])
    df = dtest.drop(['col3'],axis = 1)
    for k in range(1,200):
        #df.loc[:,"h_x"] = theta[0]*1 + theta[1]*df.loc[0,0] + theta[2]*df.loc[0,1]
        df.loc[:,"h_x"] = df.apply(lambda x: a(x, theta),axis=1)
        print(df.loc[:,'h_x'])
        num=df.shape[0]
        print(num)
        print(list(range(len(theta))))
        print(df)
        
        ax1 = (df.loc[:,'h_x'] - output).sum()
        theta[0]= theta[0] - ax1*alpha/num

        for i in range(1,len(theta)):
            ax = df.loc[:,"h_x"]*df.iloc[:,i-1]
            bx = output*df.iloc[:,i-1]
            sum1 = (ax-bx).sum()
            theta[i] = theta[i]- sum1*alpha/num
            
    return theta
   # print('theta',theta)
    
# BEGIN: from https://github.com/Shitao/Kaggle-House-Prices-Advanced-Regression-Techniques/blob/master/code/ensemble/ensemble_with_best.py
    # EXPLANATION: Use another model or "stacker" (which is Ridge here), to combine all previous model predictions in order to reduce generalization errors

class ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self,train,test,ytr):
        X = train.values
        y = ytr.values
        T = test.values
        folds = list(KFold(len(y), n_folds = self.n_folds, shuffle = True, random_state = 0))
        S_train = np.zeros((X.shape[0],len(self.base_models)))
        S_test = np.zeros((T.shape[0],len(self.base_models))) # X need to be T when do test prediction
        for i,reg in enumerate(self.base_models):
            print ("Fitting the base model...")
            S_test_i = np.zeros((T.shape[0],len(folds))) # X need to be T when do test prediction
            for j, (train_idx,test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                reg.fit(X_train,y_train)
                y_pred = reg.predict(X_holdout)[:]
                S_train[test_idx,i] = y_pred
                S_test_i[:,j] = reg.predict(T)[:]
            #    S_test_i[:,j] = reg.predict(X)[:]
            S_test[:,i] = S_test_i.mean(1)

        
        print ("Stacking base models...")
        param_grid = {
	     'alpha': [0.001,0.01,0.1,0.5,0.8,1],
	}
        grid = model_selection.GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, scoring='neg_mean_squared_error')
        grid.fit(S_train, y)
        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
        except:
                pass

        y_pred = grid.predict(S_test)[:]
        return y_pred, -grid.best_score_

 # END: from https://github.com/Shitao/Kaggle-House-Prices-Advanced-Regression-Techniques/blob/master/code/ensemble/ensemble_with_best.py
    


#TAKING THE MEAN OF THE PREDICTIONS OF THE BEST MODELS
def meanpred():
    a = pd.read_csv('xgbr.csv')
    b = pd.read_csv('gbr.csv')
    c = pd.read_csv('stacked.csv')
    
    d = a.loc[:,'SalePrice'] + b.loc[:,'SalePrice'] + c.loc[:,'SalePrice']
    pred = d/3
    writefile(pred,a,'average.csv')
 
    
    
#Standardizing the outputSeries
def outputSTD(outputSeries):
    mean = outputSeries.mean()
    deviation = outputSeries.std()
    outputSeries = outputSeries - mean
    outputSeries = outputSeries/deviation
    return outputSeries,mean,deviation


#main
def main():
    trainDF,testDF = readData()
    train = preprocessing(trainDF,trainDF)
    test = preprocessing(testDF,trainDF)
    train1 = outliers2(trainDF)
    train = outliers2(train)
    
    #get_dummies function to manipulate the categorical attributes to prepare to run the model
    print(train.shape[0])
    print(test.shape[0])
    df = pd.concat([train,test])
    df = pd.get_dummies(df)
    inputCols = df.columns
   
    standardize(df, inputCols)
    
    train = df.iloc[:train.shape[0],:]
    test = df.iloc[train.shape[0]:,:]
    
    #Check the size of the training and testing data set
    print(train.shape[0])
    print(test.shape[0])
    
    
    inputCols = train.columns
   # print(trainDF)
    outputSeries = train1.loc[:,'SalePrice']
    outputstd,meanout,deviout = outputSTD(outputSeries)

    
    
#USING GRADIENT DESCENT
    print()
    print("USING GRADIENT DESCENT")
    theta = gradientDescent(train, outputstd)
    print('Done calculating theta for Gradident Descent')
    out= pd.Series(test.apply(lambda x: a(x, theta),axis=1))
    
    #Reverse Standardization
    out = out*deviout + meanout
    print(out)
    
    writefile(out,testDF,'gradscent.csv')
    print('Done writing Gradscen.csv')
    '''
    
    
    #generate predictions with untuned GBR
    '''
    #create an object of GBR without tuning and write the prediction file
    gbr = GradientBoostingRegressor()
    gbr.fit(train.loc[:,inputCols], outputSeries)
    #print(train.isnull().any())
    print(model_selection.cross_val_score(gbr,train.loc[:,inputCols], outputSeries,
                                          cv=10, scoring='neg_mean_squared_error').mean())
    predictions = gbr.predict(test)
    print(predictions)
    print(len(predictions))
    print('Done predictions for untuned GBR')
    
    
#PARAMETER TUNING
    #testing how parameter tuning works
    '''
    print('Test n_estimators')
    param_test1 = {'n_estimators':list(range(20,10000,1000))}
    gsearch1 = model_selection.GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.05, 
                                                                                  min_samples_split=500,
                                                                                  min_samples_leaf=50,
                                                                                  max_depth=8,
                                                                                  max_features='sqrt',
                                                                                  subsample=0.8,
                                                                                  random_state=10), 
                                            param_grid = param_test1, 
                                            scoring='neg_mean_squared_error',
                                            n_jobs=44,
                                            iid=False, 
                                            cv=5)
    
    #fitting the model
    gsearch1.fit(train.loc[:,inputCols],outputSeries)
    
    #print out the best scores and parameters 
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    print('Done testing n_estimators for GBR')
    '''
    
    #Parameters Tuning for GBR
    '''
    gbr = GradientBoostingRegressor()
    print()
    print('GBR Parameter Tuning')
    param_test2 = {
            'n_estimators': [1000],
            'min_samples_split': np.arange(2,10),
            'max_depth':np.arange(2,8), 
            'max_features':['sqrt'],
            'learning_rate': [0.01],
            'subsample': [0.7,0.75,0.8,0.85,0.9]
            }
    model = model_selection.GridSearchCV(estimator =gbr, 
                                         param_grid = param_test2,
                                         n_jobs = 44, 
                                         cv=10, 
                                         scoring = 'neg_mean_squared_error')
    
    model.fit(train.loc[:,inputCols],outputSeries)
    print('try')
    print('best param:')
    print(model.best_params_)
    print('best CV')
    print(-model.best_score_)
    print('Done tuning GBR')
    
    '''
    #create tuned_gbr model and write result file
    
    
    tuned_gbr = GradientBoostingRegressor(learning_rate=0.01, 
                                          n_estimators=1000,
                                          max_depth=4, 
                                          min_samples_split=6, 
                                          subsample=0.7, 
                                          random_state=10, 
                                          max_features='sqrt',
                                          warm_start=True)
    tuned_gbr.fit(train.loc[:,inputCols], outputSeries)
    
    #print(train.isnull().any())
    print(model_selection.cross_val_score(tuned_gbr,
                                          train.loc[:,inputCols], 
                                          outputSeries, 
                                          cv=10, 
                                          scoring='neg_mean_squared_error').mean())
    
    predictions = tuned_gbr.predict(test)
    print(predictions)
    print(len(predictions))
    
    writefile(predictions,testDF,'gbr.csv')
    
    print('Done generating predictions for tuned_GBR')
    
    #Random Forest Regressor Parameters tuning
    
    '''
    print() 
    print('RFR Parameter Tuning')
    rfr = RandomForestRegressor(n_jobs=44, random_state=0)
    param_test3 = {
            'n_estimators': [1000],
            'min_samples_split': np.arange(2,10),
            'max_depth':np.arange(2,8), 
            'max_features':['sqrt'],
            'min_samples_leaf':range(2,9)
            }
    model3 = model_selection.GridSearchCV(estimator =rfr, 
                                          param_grid = param_test3,
                                          n_jobs = 44, 
                                          cv=10, 
                                          scoring = 'neg_mean_squared_error')
    
    model3.fit(train.loc[:,inputCols],outputSeries)
    print('try')
    print('best param:')
    print(model3.best_params_)
    print('best CV')
    print(-model3.best_score_)
    print('Done tuning RFR')
    '''
    #create tuned_rfr model and write result file
        
    

    tuned_rfr = RandomForestRegressor(n_estimators=1000,
                                      max_depth=7, 
                                      min_samples_split=2,
                                      min_samples_leaf=2, 
                                      random_state=10, 
                                      max_features='sqrt',
                                      warm_start=True)
    
    tuned_rfr.fit(train.loc[:,inputCols], outputSeries)
    
    #print(train.isnull().any())
    print(model_selection.cross_val_score(tuned_rfr,
                                          train.loc[:,inputCols], 
                                          outputSeries, 
                                          cv=10, 
                                          scoring='neg_mean_squared_error').mean())
    
    predictions2 = tuned_rfr.predict(test)
    print(predictions2)
    print(len(predictions2))
    
    writefile(predictions2,testDF,'rfr.csv')
    print('Done generating predictions for RFR')
    
    
    #XGBR
    '''
    print()
    print('XGBOOST parameters tuning')
    xgbr = xgb.XGBRegressor(seed=0)
    
    param_test4 = {
            'n_estimators': [1000],
            'max_depth':list(range(2,12,2)), 
            'learning_rate': [0.01],
            'subsample': [0.7,0.75,0.8,0.85,0.9],
            'colsample_bytree': [0.7,0.8,0.9]
    }
    
    model4 = model_selection.GridSearchCV(estimator =xgbr, 
                                          param_grid = param_test4,
                                          n_jobs = 44, 
                                          cv=10, 
                                          scoring = 'neg_mean_squared_error')
    
    model4.fit(train.loc[:,inputCols],outputSeries)
    print('try')
    print('best param:')
    print(model4.best_params_)
    print('best CV')
    print(-model4.best_score_)
    print('Done tuning xgBoost')
    '''
    
    #create tuned_xgbr model and write result file
    

    tuned_xgbr = xgb.XGBRegressor(n_estimators=1000,
                                  max_depth=4,
                                  learning_rate = 0.01,
                                  subsample = 0.75,
                                  colsample_bytree=0.8)
    
    tuned_xgbr.fit(train.loc[:,inputCols], outputSeries)
    
    #print(train.isnull().any())
    print(model_selection.cross_val_score(tuned_xgbr,
                                          train.loc[:,inputCols], 
                                          outputSeries, 
                                          cv=10, 
                                          scoring='neg_mean_squared_error').mean())
    predictions3 = tuned_xgbr.predict(test)
    print(predictions3)
    print(len(predictions3))
    
    writefile(predictions3,testDF,'xgbr.csv')
    print('Done generating predictions for xgBoost')
    
    
    
    
    #extra tree regressor
    '''
    
    print()
    print('Extra tree regressor parameters tuning')
    etr = ExtraTreesRegressor(n_jobs=44, random_state=0)
    param_test5 = {
            'n_estimators': [1000],
            'max_depth':list(range(2,12,2)), 
            'min_samples_split': np.arange(2,10),
            'min_samples_leaf':range(2,9),
            'max_features':['sqrt']
            }
    model5 = model_selection.GridSearchCV(estimator =etr, 
                                          param_grid = param_test5,
                                          n_jobs = 44, 
                                          cv=10, 
                                          scoring = 'neg_mean_squared_error')
    
    model5.fit(train.loc[:,inputCols],outputSeries)
    print('try')
    print('best param:')
    print(model5.best_params_)
    print('best CV')
    print(-model5.best_score_)
    print('Done tuning ETR')
    '''
    #create tuned_etr model and write result file
    
    tuned_etr = ExtraTreesRegressor(n_estimators=1000,
                                    max_depth=10, 
                                    min_samples_split=2,
                                    min_samples_leaf=2, 
                                    random_state=10, 
                                    max_features='sqrt',
                                    warm_start=True)
    
    tuned_etr.fit(train.loc[:,inputCols], outputSeries)
    
    #print(train.isnull().any())
    
    print(model_selection.cross_val_score(tuned_etr,
                                          train.loc[:,inputCols], 
                                          outputSeries, 
                                          cv=10, 
                                          scoring='neg_mean_squared_error').mean())
    
    predictions4 = tuned_etr.predict(test)
    print(predictions4)
    print(len(predictions4))
    
    writefile(predictions4,testDF,'etr.csv')
    print('Done generating predictions for ETR')
    
    print('done')
    
    
    #USING THE ENSEMBLE CLASS TO GENERATE PREDICTION FOR STACKING ALL PREVIOUS MODELS
    
    base_models = [tuned_gbr,tuned_rfr,tuned_xgbr,tuned_etr]
    
    ensem = ensemble(
            n_folds=5,
            stacker = Ridge(),
            base_models = base_models
        )
    
    print(train.shape)
    print(test.shape)
    print(len(outputSeries))
    y_pred, score = ensem.fit_predict(train,test,outputSeries)
    
    writefile(y_pred,testDF,"stacked.csv")
    print('Done generating predictions using Stack Models')
    
    meanpred()
    print('Done generating predictions by computing the mean of the best models')
main()
