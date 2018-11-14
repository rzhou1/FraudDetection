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

    def _feature_device_id_unique_users(self, data):
        ''' 'device_id_unique_users': how many unique users use the same device_id?'''
        deviceIdUsers = data.groupby('device_id')['user_id'].nunique()
        deviceIdUsers = pd.DataFrame(deviceIdUsers).rename(columns={'device_id': 'device_id', 'user_id': 'device_id_unique_users'})
        data = pd.merge(data, deviceIdUsers, how='left', on='device_id')
        return data


    def _feature_total_avg_purchase(self, data):
        '''create features total_purchase and avg_purchase.'''
        totalPurchase = data.groupby('device_id')['purchase_value'].agg({'total_purchase': 'sum'})
        totalPurchase = pd.DataFrame(totalPurchase)
        data = pd.merge(data, totalPurchase, how='left', on='device_id')
        data['avg_purchase'] = data['total_purchase'] / data['device_id_unique_users']
        return data

    def _feature_country_count(self, data):
        '''create features country_count: how many counts from the same country.'''
        countryCount = data.groupby('country')['user_id'].agg({'country_count': 'size'})
        countryCount = pd.DataFrame(countryCount)
        data = pd.merge(data, countryCount, how='left', on='country')
        return data

    def _feature_time_diff(self, data):
        '''create feature time_diff: time difference between purchase and signup times.'''
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])
        data['signup_time'] = pd.to_datetime(data['signup_time'])
        data['time_diff'] = data['purchase_time'] - data['signup_time']
        data['time_diff'] = data['time_diff'] / np.timedelta64(1, 's')
        return data

    def _feature_ip_users(self, data):
        '''create feature 'ip_users': how many users use the same ip address.'''
        ipUsers = data.groupby('ip_address')['user_id'].nunique()
        ipUsers = pd.DataFrame(ipUsers).rename(columns={'user_id': 'ip_users'})
        data = pd.merge(data, ipUsers, how='left', on='ip_address')
        return data

    def _feature_DOTW_WOTY(self, data):
        '''create features 'day_of_the_week, DOTW' and 'week_of_the_year, WOTY' from both signup and purchase dates.'''
        data['DOTW_signup'] = pd.to_datetime(data['signup_time']).dt.day_name()
        data['DOTW_purchase'] = pd.to_datetime(data['purchase_time']).dt.day_name()
        data['WOTY_signup'] = pd.to_datetime(data['signup_time']).dt.weekofyear
        data['WOTY_purchase'] = pd.to_datetime(data['purchase_time']).dt.weekofyear
        return data
        
    def transform(self, data):
        data = self._feature_device_id_unique_users(data)
        data = self._feature_total_avg_purchase(data)
        data = self._feature_country_count(data)
        data = self._feature_time_diff(data)
        data = self._feature_ip_users(data)
        data = self._feature_DOTW_WOTY(data)
        return data
        

class Preprocess(object):
    def __init__(self):
        pass

    def _feature_drop(self, data):
        colsDropped = ['signup_time', 'purchase_time', 'lower_bound_ip_address', 'upper_bound_ip_address']
        data = data.drop(colsDropped, axis=1)
        return data

    def _imputer(self, data):
        '''imputer the missing values:  country: 21966; country_count: 21966.'''
        data.loc[data['country'].isnull(), 'country'] = 'unknown'
        data.loc[data['country_count'].isnull(), 'country_count'] = 0
        return data

    def _bin_country(self, data):
        data.loc[data['country_count'] < 200, 'country'] = 'Small'  # 50 countries with counts > 200.
        return data

    def _log_transform(self, data):
        '''Log transform to reduce data skewness of numeric features.'''
        numericFeatures = data.select_dtypes(include='number').columns.tolist()
        numericFeatures.remove('class')
        skewness = data[numericFeatures].apply(lambda x: skew(x.dropna())).sort_values()
        skewFeatures = skewness[abs(skewness) > 0.5].index
        for feature in skewFeatures:
            data[feature] = np.log1p(data[feature])
        return data

    def _feature_encoding(self, data):
        le = preprocessing.LabelEncoder().fit(data['device_id'])
        data['device_id'] = le.transform(data['device_id'])
        return data

    def _categorical_feature_encoding(self, data):
        data = pd.get_dummies(data)
        return data

    def preprocess(self, data):
        data = self._feature_drop(data)
        data = self._imputer(data)
        data = self._bin_country(data)
        data = self._log_transform(data)
        data = self._feature_encoding(data)
        data = self._categorical_feature_encoding(data)
        return data

#train and test data split
def trainTestSplit(data, test_size, random_state):
    cols = data.columns.tolist()
    cols = cols[0:5] + cols[6:-1] + cols[5:6]
    data = data[cols]

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

#modeling
def modeling(x_train, y_train, x_test, y_test):
    metricsResults = []
    models = []
    tprResults = []
    fprResults = []
    predResults = []
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini', max_features='log2',
                                min_samples_split=3,random_state=0, n_jobs=-1)
    xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                             booster='gbtree', reg_alpha=0, reg_lambda=1, random_state=0, n_jobs=4)
    classifiers = [rf, xgbc]
    classifiersName = ['RandomForest', 'XGBoost']
    metricsTerminology = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix']

    for i in range(len(classifiers)):
        model = classifiers[i]
        model.fit(x_train, y_train)
        models.append(model)
        y_hat = model.predict(x_test)
        predResults.append(y_hat)
        y_hat_proba = model.predict_proba(x_test)[:, 1]
        tpr, fpr, thresholds = roc_curve(y_test, y_hat_proba)
        tprResults.append(tpr)
        fprResults.append(fpr)
        metricsResults.append((accuracy_score(y_test, y_hat), precision_score(y_test, y_hat), recall_score(y_test, y_hat),
             f1_score(y_test, y_hat), auc(tpr, fpr), confusion_matrix(y_test, y_hat)))

    metricsResults = pd.DataFrame({'Metrics': metricsTerminology, '{}'.format(classifiersName[0]): metricsResults[0],
                                   '{}'.format(classifiersName[1]): metricsResults[1]})
    featureImportances = pd.DataFrame({'Features': x_train.columns.tolist(), '{}'.format(classifiersName[0]): models[0].feature_importances_,
         '{}'.format(classifiersName[1]): models[1].feature_importances_}).sort_values('{}'.format(classifiersName[1]), ascending=False).head(15)

    return metricsResults, models, predResults, tprResults, fprResults, featureImportances


def plotROC(tpr, fpr):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], c='k', linestyle='--')
    plt.plot(tpr[0], fpr[0], label='RandomForest AUC: {:.4f}'.format(auc(tpr[0], fpr[0])), c='g', linestyle='-')
    plt.plot(tpr[1], fpr[1], label='XGBoost AUC: {:.4f}'.format(auc(tpr[1], fpr[1])), c='b', linestyle='-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Fraud detection - Receiver Operating Characterisics')
    plt.legend(loc='lower right')
    plt.plot()

if __name__ == '__main__':
    data = loadMergeData()
    data = FeatureExtraction().transform(data)
    data = Preprocess().preprocess(data)
    x_train, x_test, y_train, y_test = trainTestSplit(data, test_size==0.3, random_state==0)
    metricsResults, models, predResults, tprResults, fprResults, featureImportances = modeling(x_train, y_train, x_test, y_test)
    plotROC(tprResults, fprResults)
    print metricsResults
    print featureImportances



