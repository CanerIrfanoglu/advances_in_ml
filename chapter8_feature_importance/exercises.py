from sklearn.datasets import make_classification
import pandas as pd
import datetime
import numpy as np
# 8.1 Using thecode presented in Section 8.6:
# (a) Generate adataset (X,y).
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem
    trnsX_data, cont_data = make_classification( # Renamed to avoid conflict with DataFrame trnsX
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=0,
        shuffle=False
    )

    # 1. Create the DatetimeIndex and store it in df0_index
    df0_index = pd.date_range(
        end=datetime.datetime.today(),
        periods=n_samples,
        freq=pd.tseries.offsets.BDay() # Or freq='B'
    )

    # 2. Use df0_index when creating the DataFrame and Series
    # Also, note the tuple assignment: trnsX, cont = (DataFrame(...), Series(...).to_frame())
    # It's clearer to assign them separately if they don't depend on each other in one line.
    trnsX = pd.DataFrame(trnsX_data, index=df0_index)
    cont = pd.Series(cont_data, index=df0_index).to_frame('bin')

    # 3. Create column names and store them in a separate variable
    # Use range() instead of xrange()
    feature_column_names = ['I_' + str(i) for i in range(n_informative)] + \
                           ['R_' + str(i) for i in range(n_redundant)]
    feature_column_names += ['N_' + str(i) for i in range(n_features - len(feature_column_names))]
    trnsX.columns = feature_column_names

    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index) # This creates a column t1 with the index values
    return trnsX, cont

X, y_df = getTestData(n_samples=5000)

# (b) Apply a PCA transformation on X, which we denotė X.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=X.shape[1])  # retain all components
X_pca = pd.DataFrame(pca.fit_transform(X), index=X.index)
X_pca.columns = X.columns

# Step 3: Optional - check explained variance
explained_variance_ratio = pca.explained_variance_ratio_

# plt.figure(figsize=(10, 5))
# plt.plot(np.cumsum(explained_variance_ratio))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by PCA Components')
# plt.grid(True)
# plt.show()

# (c) Compute MDI, MDA, and SFI feature importance on (X_pca,y), where the base estimator is RF.
# feature importance from a random forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold

# We want to run featImportance here. All below functions are its preprequisites

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

            # CAI: Below code throwing a warning replacing with the iloc versions below
            # maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            # train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)

            maxT1Idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)

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
        
        score.append(score_)
    
    return np.array(score)

def featImpMDI(fit,featNames):
    # feat importance based on IS mean impurity reduction
    df0={i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # because max_features=1
    imp=pd.concat({'mean':df0.mean(),'std':df0.std()*df0.shape[0]**-.5},axis=1)
    imp/=imp['mean'].sum()
    return imp
         
def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv
    scr0,scr1=pd.Series(),pd.DataFrame(columns=X.columns)
    
    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0=X.iloc[train,:],y.iloc[train],sample_weight.iloc[train]
        X1,y1,w1=X.iloc[test,:],y.iloc[test],sample_weight.iloc[test]
        fit=clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1)
            scr0.loc[i]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
        else:
            pred=fit.predict(X1)
            scr0.loc[i]=accuracy_score(y1,pred,sample_weight=w1.values)
        for j in X.columns:
            X1_=X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of a single column
            if scoring=='neg_log_loss':
                prob=fit.predict_proba(X1_)
                scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
            else:
                pred=fit.predict(X1_)
                scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
    
    imp=(-scr1).add(scr0,axis=0)
    if scoring=='neg_log_loss':
        imp=imp/-scr1 
    else:
        imp=imp/(1.-scr1)

    imp=pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},axis=1)
    return imp,scr0.mean()


def auxFeatImpSFI(featNames,clf,trnsX,cont,scoring,cvGen):
    imp=pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        df0=cvScore(clf,X=trnsX[[featName]],y=cont['bin'],sample_weight=cont['w'],scoring=scoring,cvGen=cvGen)
        imp.loc[featName,'mean']=df0.mean()
        imp.loc[featName,'std']=df0.std()*df0.shape[0]**-.5
    return imp


# using this instead of mpPandasObj
def run_feat_imp_seq(feat_names, clf, trnsX, cont, scoring, cvGen):
    """
    Sequential fallback for mpPandasObj when testing.
    """
    output = []
    from tqdm import tqdm
    for feat in tqdm(feat_names, desc='Calculating SFI'):
        res = auxFeatImpSFI(featNames=[feat], clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
        output.append(res)
    return pd.concat(output, axis=0)



from sklearn.ensemble import RandomForestClassifier

def featImportance(trnsX,cont,n_estimators=100,cv=10,max_samples=1.,numThreads=24,
    pctEmbargo=0,scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    n_jobs=(-1 if numThreads>1 else 1) # run 1 thread with ht_helper in dirac1
    
    # # CAI: Comment out below and use randomforest to comply with exercise requirement
    # # 1) prepare classifier,cv. max_features=1, to prevent masking
    # clf=DecisionTreeClassifier(criterion='entropy',max_features=1,
    # class_weight='balanced',min_weight_fraction_leaf=minWLeaf)
    # clf=BaggingClassifier(estimator=clf,n_estimators=n_estimators,max_features=1.,max_samples=max_samples,oob_score=True,n_jobs=n_jobs)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features='sqrt',  # or use default
        class_weight='balanced',
        min_weight_fraction_leaf=minWLeaf,
        n_jobs=n_jobs,
        criterion='entropy'
    )
    
    fit=clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values)
    # oob=fit.oob_score_
    oob=fit.oob_score
    if method=='MDI':
        imp=featImpMDI(fit,featNames=trnsX.columns)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring).mean()
    elif method=='MDA':
        imp,oos=featImpMDA(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],
        t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring)
    elif method=='SFI':
        cvGen=PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],scoring=scoring,cvGen=cvGen).mean()
        clf.n_jobs=1 # paralellize auxFeatImpSFI rather than clf
        # parallel processing function mpPandasObj is not available instead using below
        # imp=mpPandasObj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen)
        imp = run_feat_imp_seq(trnsX.columns, clf=clf, trnsX=trnsX, cont=cont, scoring='accuracy', cvGen=cvGen)

    imp = imp.sort_values(by='mean', ascending=False)

    return imp,oob,oos


# MDI
imp_mdi, oob_mdi, oos_mdi = featImportance(
    trnsX=X_pca, cont=y_df, method='MDI', scoring='accuracy', cv=10, numThreads=1
)

print(imp_mdi)
print('---')
print(oob_mdi)
print(oos_mdi)

# MDA
imp_mda, oob_mda, oos_mda = featImportance(
    trnsX=X_pca, cont=y_df, method='MDA', scoring='accuracy', cv=10, numThreads=1
)

print(imp_mda)
print('---')
print(oob_mda)
print(oos_mda)

# SFI
imp_sfi, oob_sfi, oos_sfi = featImportance(
    trnsX=X_pca, cont=y_df, method='SFI', scoring='accuracy', cv=10, numThreads=1
)

print(imp_sfi)
print('---')
print(oob_sfi)
print(oos_sfi)

# (d) Do thethree methods agree on what features areimportant? Why?


# 8.2 From exercise 1, generate a new dataset (X_fu,y), where # X is a feature union of X # anḋ X_pca
X_fu = pd.concat([X, X_pca], axis=1)
# MDI
imp_mdi_fu, oob_mdi_fu, oos_mdi_fu = featImportance(
    trnsX=X_fu, cont=y_df, method='MDI', scoring='accuracy', cv=10, numThreads=1
)

print(imp_mdi_fu)
print('---')
print(oob_mdi_fu)
print(oos_mdi_fu)

# MDA
# Getting 0 for all here because now all features have their identical ones, cancelling each other
imp_mda_fu, oob_mda_fu, oos_mda_fu = featImportance(
    trnsX=X_fu, cont=y_df, method='MDA', scoring='accuracy', cv=10, numThreads=1
)

print(imp_mda_fu)
print('---')
print(oob_mda_fu)
print(oos_mda_fu)

# SFI
imp_sfi_fu, oob_sfi_fu, oos_sfi_fu = featImportance(
    trnsX=X_fu, cont=y_df, method='SFI', scoring='accuracy', cv=10, numThreads=1
)

print(imp_sfi)
print('---')
print(oob_sfi)
print(oos_sfi)
