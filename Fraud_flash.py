'''Solution for fraudulent activities detection by using both tree-based models and logistic regressions.
The solution also evaluates using the original full dataset and separating original dataset into non-flash-transaction dataset and flash-transaction dataset,
where the latter was discovered having unity fraud during EDA and modeling.
The combined performance from model from non-flash-transaction and rule of flash-transaction is almost identical as that of the solution with full dataset,
suggesting that the original data is not sufficient to catch all frauds.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf
from scipy.stats import skew
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, precision_score, recall_score, f1_score, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#load data
def loadMergeData():
    '''Load and merge raw data sets.'''
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_data = pd.read_csv('IpAddress_to_Country.csv')

    pysqldf = lambda q: sqldf(q, globals())#Merge fraud_data and ip_data by pandas sql
    data = pysqldf("""SELECT * 
        FROM 
            fraud_data f
        LEFT JOIN
            ip_data i
        ON f.ip_address >= i.lower_bound_ip_address AND f.ip_address <=i.upper_bound_ip_address""")
    return data

#Feature extraction from raw dataset
class FeatureExtraction(object):
    '''Feature engineering: features extraction from raw dataset.'''
    def __init__(self):
        pass

    def feature_device_id_unique_users(self, data):
        ''' 'device_id_unique_users': how many unique users use the same device_id?'''
        deviceIdUsers = data.groupby('device_id')['user_id'].nunique()
        deviceIdUsers = pd.DataFrame(deviceIdUsers).rename(columns={'device_id': 'device_id', 'user_id': 'device_id_unique_users'})
        data = pd.merge(data, deviceIdUsers, how='left', on='device_id')
        return data


    def feature_total_avg_purchase(self, data):
        '''create features total_purchase and avg_purchase.'''
        totalPurchase = data.groupby('device_id')['purchase_value'].agg({'total_purchase': 'sum'})
        totalPurchase = pd.DataFrame(totalPurchase)
        data = pd.merge(data, totalPurchase, how='left', on='device_id')
        data['avg_purchase'] = data['total_purchase'] / data['device_id_unique_users']
        return data

    def feature_country_count(self, data):
        '''create features country_count: how many counts from the same country.'''
        countryCount = data.groupby('country')['user_id'].agg({'country_count': 'size'})
        countryCount = pd.DataFrame(countryCount)
        data = pd.merge(data, countryCount, how='left', on='country')
        return data

    def feature_time_diff(self, data):
        '''create feature time_diff: time difference between purchase and signup times.'''
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])
        data['signup_time'] = pd.to_datetime(data['signup_time'])
        data['time_diff'] = data['purchase_time'] - data['signup_time']
        data['time_diff'] = data['time_diff'] / np.timedelta64(1, 's')
        return data

    def feature_ip_users(self, data):
        '''create feature 'ip_users': how many users use the same ip address.'''
        ipUsers = data.groupby('ip_address')['user_id'].nunique()
        ipUsers = pd.DataFrame(ipUsers).rename(columns={'user_id': 'ip_users'})
        data = pd.merge(data, ipUsers, how='left', on='ip_address')
        return data

    def feature_DOTW_WOTY(self, data):
        '''create features 'day_of_the_week, DOTW' and 'week_of_the_year, WOTY' from both signup and purchase dates.'''
        data['DOTW_signup'] = pd.to_datetime(data['signup_time']).dt.day_name()
        data['DOTW_purchase'] = pd.to_datetime(data['purchase_time']).dt.day_name()
        data['WOTY_signup'] = pd.to_datetime(data['signup_time']).dt.weekofyear
        data['WOTY_purchase'] = pd.to_datetime(data['purchase_time']).dt.weekofyear
        return data
        
    def transform(self, data):
        data = self.feature_device_id_unique_users(data)
        data = self.feature_total_avg_purchase(data)
        data = self.feature_country_count(data)
        data = self.feature_time_diff(data)
        data = self.feature_ip_users(data)
        data = self.feature_DOTW_WOTY(data)
        return data
        

class Preprocess(object):
    def __init__(self):
        pass

    def feature_drop(self, data):
        colsDropped = ['signup_time', 'purchase_time', 'lower_bound_ip_address', 'upper_bound_ip_address', 'avg_purchase', 'purchase_value', 'ip_users','WOTY_signup', 'DOTW_signup']
        data = data.drop(colsDropped, axis=1)
        return data

    def imputer(self, data):
        '''imputer the missing values:  country: 21966; country_count: 21966.'''
        data.loc[data['country'].isnull(), 'country'] = 'unknown'
        data.loc[data['country'] == 'unknown', 'country_count'] = 21966  # data_all.loc[data_all.country=='unknown', :].count()
        data.loc[data['country_count'].isnull(), 'country_count'] = 0
        return data

    def bin_country(self, data):
        data.loc[data['country_count'] < 200, 'country'] = 'Small'  # 50 countries with counts > 200.
        return data

    def reduce_skewness(self, data):
        '''Log transform to reduce data skewness of numeric features.'''
        numericFeatures = data.select_dtypes(include='number').columns.tolist()
        numericFeatures.remove('class')
        skewness = data[numericFeatures].apply(lambda x: skew(x.dropna())).sort_values()
        skewFeatures = skewness[abs(skewness) > 0.5].index
        for feature in skewFeatures:
            data[feature] = np.log1p(data[feature])
        return data

    def log_transform(self, data):
        '''Non-skewed numeric features, for logistic regression.'''
        features = ['age', 'time_diff', 'country_count']

        for feature in features:
            data[feature] = np.log1p(data[feature])

        return data

    def convert_dtypes(self, data):

        data['WOTY_purchase'] = data['WOTY_purchase'].astype(str)

        return data

    def feature_encoding(self, data):
        le = preprocessing.LabelEncoder().fit(data['device_id'])
        data['device_id'] = le.transform(data['device_id'])
        return data

    def categorical_feature_encoding(self, data):
        data = pd.get_dummies(data)
        return data

    def preprocess(self, data):
        data = self.feature_drop(data)
        data = self.imputer(data)
        data = self.bin_country(data)
        data = self.reduce_skewness(data)
        data = self.log_transform(data)
        data = self.convert_dtypes(data)
        data = self.feature_encoding(data)
        data = self.categorical_feature_encoding(data)
        return data

#train, val and test data split
def trainTestSplit(data, random_state):
    cols = data.columns.tolist()
    cols = cols[0:2] + cols[3:4] + cols[2:3] + cols[5:] + cols[4:5]
    data = data[cols]

    test_val_size = 0.3
    test_size = 0.33

    X = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=test_val_size,
                                                                random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size,
                                                    random_state=random_state)

    X_train_n1s = X_train.loc[X_train['time_diff'] != np.log1p(1), :]
    y_train_n1s = y_train[X_train_n1s.index]
    X_val_n1s = X_val.loc[X_val['time_diff'] != np.log1p(1), :]
    y_val_n1s = y_val[X_val_n1s.index]
    X_test_n1s = X_test.loc[X_test['time_diff'] != np.log1p(1), :]
    y_test_n1s = y_test[X_test_n1s.index]

    X_train_1s = X_train.loc[X_train['time_diff'] == np.log1p(1), :]
    y_train_1s = y_train[X_train_1s.index]
    X_val_1s = X_val.loc[X_val['time_diff'] == np.log1p(1), :]
    y_val_1s = y_val[X_val_1s.index]
    X_test_1s = X_test.loc[X_test['time_diff'] == np.log1p(1), :]
    y_test_1s = y_test[X_test_1s.index]

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_n1s, X_val_n1s, X_test_n1s, y_train_n1s, y_val_n1s,\
           y_test_n1s, X_train_1s, X_val_1s, X_test_1s, y_train_1s, y_val_1s, y_test_1s


#modeling
def model_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    metrics = []
    tprs = []
    fprs = []
    probs = []
    preds = []

    xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                             booster='gbtree', reg_alpha=0, reg_lambda=1, random_state=0, n_jobs=4)
    metricsTerminology = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix']

    xgbc.fit(X_train, y_train)
    val_test = ((X_val, y_val), (X_test, y_test))
    for i, (X, y) in enumerate(val_test):
        y_hat_proba = xgbc.predict_proba(X)[:, 1]
        probs.append(y_hat_proba)
        y_hat = (y_hat_proba>0.1)
        preds.append(y_hat)
        tpr, fpr, thresholds = roc_curve(y, y_hat_proba)
        tprs.append(tpr)
        fprs.append(fpr)
        metrics.append((accuracy_score(y, y_hat), precision_score(y, y_hat), recall_score(y, y_hat), f1_score(y, y_hat),
                           auc(tpr, fpr), confusion_matrix(y, y_hat)))


    metricsResults = pd.DataFrame({'Metrics': metricsTerminology, 'val': metrics[0], 'test': metrics[1]})
    featureImportances = pd.DataFrame({'Features': X_train.columns.tolist(), 'XGBoost': xgbc.feature_importances_}).\
        sort_values('XGBoost', ascending=False).head(20)

    return probs, preds, tprs, fprs, metricsResults, featureImportances

def combineMetrics(val_test_ys):
    '''Combine predicted probability from non-flash-transaction and hold-out flash-transaction targets (1 for all observations).'''
    y_hat_proba_val_n1s, y_hat_proba_test_n1s, y_val_n1s, y_test_n1s, y_val_1s, y_test_1s = val_test_ys

    y_val_1s, y_test_1s = np.array(y_val_1s), np.array(y_test_1s)

    y_val = np.hstack((y_val_n1s, y_val_1s))
    y_test = np.hstack((y_test_n1s, y_test_1s))

    y_hat_val = np.hstack((y_hat_proba_val_n1s, y_val_1s))
    y_hat_test = np.hstack((y_hat_proba_test_n1s, y_test_1s))

    tpr_val, fpr_val, threshold_val = roc_curve(y_val, y_hat_val)
    tpr_test, fpr_test, threshold_test = roc_curve(y_test, y_hat_test)

    return tpr_val, fpr_val, tpr_test, fpr_test

#ROC plot
def plotROC(tpr, fpr, tpr_c, fpr_c, tpr_n1s, fpr_n1s, dataset_name):
    '''Plot ROC curve'''
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], c='k', linestyle='--')
    plt.plot(tpr, fpr, label='Original, AUC: {:.4f}'.format(auc(tpr, fpr)), c='g', linestyle='-')
    plt.plot(tpr_c, fpr_c, label='n1s & 1s, AUC: {:.4f}'.format(auc(tpr_c, fpr_c)), c='b', linestyle='-')
    plt.plot(tpr_n1s, fpr_n1s, label='Non_flash_transaction, AUC: {:.4f}'.format(auc(tpr_n1s, fpr_n1s)), c='m', linestyle='-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Characterisics: {}'.format(dataset_name))
    plt.legend(loc='lower right')
    plt.plot()

if __name__ == '__main__':
    data = loadMergeData() #load data
    data = FeatureExtraction().transform(data) #extracting features
    data = Preprocess().preprocess(data) #data processing

    #train, validation, test data split
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_n1s, X_val_n1s, X_test_n1s, y_train_n1s, y_val_n1s, \
    y_test_n1s, X_train_1s, X_val_1s, X_test_1s, y_train_1s, y_val_1s, y_test_1s = trainTestSplit(data, 0)

    #model by XGBoost
    #model on original dataset
    probs, preds, tprs, fprs, metricsResults, featureImportances = model_xgboost(X_train, y_train, X_val, y_val, X_test,
                                                                                 y_test)
    #model on non-flash-transaction dataset
    probs_n1s, preds_n1s, tprs_n1s, fprs_n1s, metricsResults_n1s, featureImportances_n1s = model_xgboost(X_train_n1s,
                                                                                                         y_train_n1s,
                                                                                                         X_val_n1s,
                                                                                                         y_val_n1s,
                                                                                                         X_test_n1s,
                                                                                                         y_test_n1s)
    #combine predictions from non-flash-transaction and targets of flash-transaction
    tpr_val_c, fpr_val_c, tpr_test_c, fpr_test_c = combineMetrics((probs_n1s[0], probs_n1s[1], y_val_n1s, y_test_n1s, y_val_1s, y_test_1s))

    #plotROC
    #validation
    plotROC(tprs[0], fprs[0], tpr_val_c, fpr_val_c, tprs_n1s[0], fprs_n1s[0], 'Validation')
    #test
    plotROC(tprs[1], fprs[1], tpr_test_c, fpr_test_c, tprs_n1s[1], fprs_n1s[1], 'Test')

    print metricsResults  #original dataset
    print metricsResults_n1s  #non-flash_transaction
    print featureImportances #original dataset
    print featureImportances_n1s  #non-flash_transaction



