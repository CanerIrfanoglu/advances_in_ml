# 7.1 Why is shuffling a dataset before conducting k-fold CV generally a bad idea in
# finance?
# What is the purpose of shuffling? Why does shuffling defeat the purpose
# of k-foldCV infinancial datasets?

# This is a bad idea because the order of the data is important in financial datasets.

# 7.2 Take a pair of matrices (X,y), representing observed features and labels.
#  These could be one of thedatasets derived fromthe exercises in Chapter 3.
import pandas as pd
import numpy as np

# (a) Derive the performance from a 5-fold CV of an RF classifier on (X,y),
# with-out shuffling.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

X = pd.read_csv('./data/X_chapter4_new.csv', index_col=0)
X.index = pd.to_datetime(X.index)
y = pd.read_csv('./data/y_chapter4_new.csv', index_col=0)
y.index = pd.to_datetime(y.index)


import pandas as pd

# Load t1 with index as datetime
t1 = pd.read_csv('./data/barrier_touched.csv', index_col=0, parse_dates=True)
t1 = t1.squeeze()  # Converts DataFrame to Series
t1 = pd.to_datetime(t1)  # Ensure values are datetime if needed


common_idx = X.index.intersection(t1.index)

# Step 2: Align both X and t1 to the common indices
X = X.loc[common_idx]
t1 = t1.loc[common_idx]

rf = RandomForestClassifier()
kfold = KFold(n_splits=5, shuffle=False)
cv_scores_initial = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')

print('cv_scores initial =', cv_scores_initial)

# (b) Derivetheperformancefroma5-foldCVofanRFon(X,y),withshuffling.
kfold = KFold(n_splits=5, shuffle=True)
cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')

print('cv_scores with shuffle =', cv_scores)


# c / d 
# Shuffled results should have higher accuracy due to information leakeage

# 7.3 Take the samepair of matrices (X,y)you used inexercise 2.
# (a) Derive the performance from a 10-fold purged CV of an RF on (X,y), with
# 1% embargo.
def getTrainTimes(t1,testTimes):
    '''
    Given testTimes, find the times of the training observations.
    t1.index: Time when the observation started.
    t1.value: Time when the observation ended.
    testTimes: Times of testing observations.
    '''
    trn=t1.copy(deep=True)
    for i,j in testTimes.iteritems():
        df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
        df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
        df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test
        trn=trn.drop(df0.union(df1).union(df2))
    return trn

def getEmbargoTimes(times,pctEmbargo):
    # Get embargo time for each bar
    step=int(times.shape[0]*pctEmbargo)
    if step==0:
        mbrg=pd.Series(times,index=times)
    else:
        mbrg=pd.Series(times[step:],index=times[:-step])
        mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
    return mbrg

class PurgedKFold(KFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo
    
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[(i[0],i[-1]+1) for i in \
        np.array_split(np.arange(X.shape[0]),self.n_splits)]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train (with embargo)
                train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None, pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
    
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob,sample_weight=sample_weight.iloc[test].values,labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred,sample_weight= sample_weight.iloc[test].values)
        
            print("score = ",score_)
        
        score.append(score_)
    
    return np.array(score)


cv_scores = cvScore(
    clf=RandomForestClassifier(),
    X=X,
    y=y,
    sample_weight = pd.Series(1, index=X.index),
    scoring='accuracy',
    t1=t1,
    cv=5,
    pctEmbargo=0.01  # 1% embargo
)

print("cv_scores with purging = ",cv_scores)

# (b) Why istheperformance lower?
# (c) Why isthisresultmore realistic?



# 7.4.
# In this chapter we have focused on one reason why k-fold CV fails in financial
# applications,namelythefactthatsomeinformationfromthetestingsetleaksinto
# the trainingset. Can you thinkof asecond reason for CV’sfailure?

# | # | Problem                            | Description                                       |
# | - | ---------------------------------- | ------------------------------------------------- |
# | 1 | **Leakage**                        | Overlapping features or labels between train/test |
# | 2 | **Non-Stationarity**               | Market regime shifts over time                    |
# | 3 | **Autocorrelation**                | Observations aren't i.i.d.                        |
# | 4 | **Label Imbalance**                | Rare events poorly sampled                        |
# | 5 | **Path Dependency**                | Some models rely on full time series paths        |
# | 6 | **Recalibration Bias**             | Real-world model updates aren't accounted for     |
# | 7 | **Label Overlap / Low Uniqueness** | Reduces true independence of training labels      |


# cv_scores initial = 
# [0.74937133 0.75859179 0.79949664 0.82466443 0.74748322]

#cv_scores with purging =  
# [0.74769489 0.751886   0.78942953 0.82466443 0.73657718]

# cv_scores with shuffle = 
# [0.80050293 0.80720872 0.8011745  0.8204698  0.80788591]