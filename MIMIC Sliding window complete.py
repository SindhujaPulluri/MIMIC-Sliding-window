#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

conn = sqlite3.connect('/Users/jwojt/Documents/research/data/mimic2018/mimic.db')

admissions = pd.read_sql('select *, (julianday(date(admittime))-julianday(date(dob)))/365.25 as age from admissions, patients where admissions.subject_id = patients.subject_id and (julianday(date(admittime))-julianday(date(dob)))/365.25 >= 65',conn)


# In[12]:


#
# Now the actual analysis
#

window_back = 24
window_forward = 24
window_width = 24
shift = 1
window_size = window_back + window_forward + window_width

# calculate los in hours
admissions['los_hrs']=admissions.apply(lambda r: (time.mktime(time.strptime(r['DISCHTIME'],'%Y-%m-%d %H:%M:%S')) - 
                 time.mktime(time.strptime(r['ADMITTIME'],'%Y-%m-%d %H:%M:%S')))/3600.0, axis=1)

# get list of patients
pts = admissions.ix[:,1].unique()

# randomly select train/test
msk = np.random.rand(len(pts)) < 0.8
train_pts=pts[msk]
test_pts=pts[~msk]
train = admissions[admissions.ix[:,1].isin(train_pts)]
test = admissions[admissions.ix[:,1].isin(test_pts)]

train_sel = train[train['los_hrs']>=window_size]

# build time sliding window for traingn data
dts = []
for i in train_sel.index[:100]:
  pt = train_sel.loc[i]  
  print('Patient:', pt.iloc[1])
  for t in range(window_back, int(pt['los_hrs'])-window_forward, shift):
    #print('Current time: ', t)
    labs = pd.read_sql("select * from labevents where hadm_id = \"" + pt[2] + 
                       "\" and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 >= " + str( t - window_back) +
                       " and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 < " + str( t ), conn)
    #print('   found labs: ',len(labs.index))
    if len(labs.index) > 0:
        # output
        flag = 0
        if ((pt['HOSPITAL_EXPIRE_FLAG'] == '1') 
            # check if died on day of discharge
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_year == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_year)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mon == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mon)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mday == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mday)
            # check if within the window
            and (time.mktime(time.strptime(pt['ADMITTIME'],'%Y-%m-%d %H:%M:%S')) + (t + window_width) * 3600 >
                 time.mktime(time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')) )
           ):
            flag = 1
            #print('yes')
               
        labs=labs.replace("",np.nan)
        labs['vn'] = labs.apply(lambda r: float(r['VALUENUM']),axis=1)
        gr = labs.groupby('ITEMID')
        #print(pd.DataFrame(gr['vn'].mean()).T)
        d = pd.DataFrame(gr['vn'].mean()).T
        d['class'] = flag
        dts.append(d)
print('concatenating data')
dt_train = pd.concat(dts, ignore_index=True) 
del dts

# impute missing values
# bad thing to do ... but let's try for now
imp = Imputer()
imp.fit(dt_train)
dt_train_imp=pd.DataFrame(imp.transform(dt_train),columns=list(dt_train.columns))

# get list of columns for independent variables
cls = list(dt.columns)
cls.remove('class')

# build models
# LR

lr = LogisticRegression()
lr.fit(dt_train_imp[cls],dt_train_imp['class'])
probs_lr = lr.predict_proba(dt_train_imp[cls])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(dt_train_imp['class'], probs_lr[:,1])
print( "LR AUC: ", auc(fpr_lr,tpr_lr))

# RF
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(dt_train_imp[cls],dt_train_imp['class'])
probs_rf = rf.predict_proba(dt_train_imp[cls])
fpr_rf, tpr_rf, thresholds_rf = roc_curve(dt_train_imp['class'], probs_rf[:,1])
print( "RF AUC: ", auc(fpr_rf,tpr_rf))


# In[ ]:


# now the test data

test_sel = test[test['los_hrs']>=window_size]

dts = []
for i in test_sel.index[:100]:
  pt = test_sel.loc[i]  
  print('Patient:', pt.iloc[1])
  for t in range(window_back, int(pt['los_hrs'])-window_forward, shift):
    #print('Current time: ', t)
    labs = pd.read_sql("select * from labevents where hadm_id = \"" + pt[2] + 
                       "\" and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 >= " + str( t - window_back) +
                       " and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 < " + str( t ), conn)
    #print('   found labs: ',len(labs.index))
    if len(labs.index) > 0:
        # output
        flag = 0
        if ((pt['HOSPITAL_EXPIRE_FLAG'] == '1') 
            # check if died on day of discharge
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_year == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_year)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mon == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mon)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mday == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mday)
            # check if within the window
            and (time.mktime(time.strptime(pt['ADMITTIME'],'%Y-%m-%d %H:%M:%S')) + (t + window_width) * 3600 >
                 time.mktime(time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')) )
           ):
            flag = 1
            #print('yes')
               
        labs=labs.replace("",np.nan)
        labs['vn'] = labs.apply(lambda r: float(r['VALUENUM']),axis=1)
        gr = labs.groupby('ITEMID')
        #print(pd.DataFrame(gr['vn'].mean()).T)
        d = pd.DataFrame(gr['vn'].mean()).T
        d['class'] = flag
        dts.append(d)
print('concatenating data')
dt_test = pd.concat(dts, ignore_index=True) 
del dts


# map test set into train set
dt_test_map = pd.DataFrame()
for c in list(dt_train_imp.columns):
    if c in list(dt_test.columns):
        dt_test_map[c] = dt_test[c]
    else:
        dt_test_map[c] = np.nan

# impute missing values
dt_test_imp=pd.DataFrame(imp.transform(dt_test_map),columns=list(dt_test_map.columns))

# apply models
# LR

probs_lr = lr.predict_proba(dt_test_imp[cls])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(dt_test_imp['class'], probs_lr[:,1])
print( "LR AUC: ", auc(fpr_lr,tpr_lr))

# RF
probs_rf = rf.predict_proba(dt_test_imp[cls])
fpr_rf, tpr_rf, thresholds_rf = roc_curve(dt_test_imp['class'], probs_rf[:,1])
print( "RF AUC: ", auc(fpr_rf,tpr_rf))



# In[ ]:




