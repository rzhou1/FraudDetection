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
fraud_data = pd.read_csv('Fraud_Data.csv')
ip_data = pd.read_csv('IpAddress_to_Country.csv')

#Merge fraud_data and ip_data by pandas sql
pysqldf = lambda q: sqldf(q, globals())

data_all = pysqldf("""SELECT * 
    FROM 
        fraud_data f
    LEFT JOIN
        ip_data i
    ON f.ip_address >= i.lower_bound_ip_address AND f.ip_address <=i.upper_bound_ip_address""")

#data_all.to_csv('fraud_ip_data.csv', index=False)

#Feature engineering: features creation based on raw dataset
#create feature 'device_id_unique_users' (how many unique users use the same device_id)
data_all_device_id_users = data_all.groupby('device_id')['user_id'].nunique()
data_all_device_id_users = pd.DataFrame(data_all_device_id_users).rename(columns={'device_id':'device_id', 'user_id':'device_id_unique_users'})
data_all = pd.merge(data_all, data_all_device_id_users, how='left', on='device_id')
#all unique users only use one device.

#create features 'total_purchase' and 'avg_purchase'
data_all_total_purchase = data_all.groupby('device_id')['purchase_value'].agg({'total_purchase': 'sum'})
data_all_total_purchase = pd.DataFrame(data_all_total_purchase)
data_all = pd.merge(data_all, data_all_total_purchase, how='left', on='device_id')
data_all['avg_purchase'] = data_all['total_purchase']/data_all['device_id_unique_users']

#create features 'country_count'
data_all_country_count = data_all.groupby('country')['user_id'].agg({'country_count':'size'})
data_all_country_count = pd.DataFrame(data_all_country_count)
data_all = pd.merge(data_all, data_all_country_count, how='left', on='country')

#create feature 'time_diff'
data_all['purchase_time'] = pd.to_datetime(data_all['purchase_time'])
data_all['signup_time'] = pd.to_datetime(data_all['signup_time'])
data_all['time_diff'] = data_all['purchase_time']-data_all['signup_time']
data_all['time_diff'] = data_all['time_diff'] / np.timedelta64(1,'s')

#create feature 'ip_users' (same ip address for different users)
data_all_ip_users = data_all.groupby('ip_address')['user_id'].nunique()
data_all_ip_users = pd.DataFrame(data_all_ip_users).rename(columns={'user_id':'ip_users'})
data_all = pd.merge(data_all, data_all_ip_users, how='left', on='ip_address')

#create features 'day_of_the_week, DOTW' and 'week_of_the_year, WOTY' from both signup and purchase dates
data_all['DOTW_signup'] = pd.to_datetime(data_all['signup_time']).dt.day_name()
data_all['DOTW_purchase'] = pd.to_datetime(data_all['purchase_time']).dt.day_name()
data_all['WOTY_signup'] = pd.to_datetime(data_all['signup_time']).dt.weekofyear
data_all['WOTY_purchase'] = pd.to_datetime(data_all['purchase_time']).dt.weekofyear

#data preprocessing
#imputer the missing values
data_all.isnull().sum()
# country: 21966; country_count: 21966
data_all.loc[data_all['country'].isnull(), 'country'] = 'unknown'
data_all.loc[data_all['country_count'].isnull(), 'country_count'] = 0

#drop features
cols_dropped = ['signup_time', 'purchase_time',  'lower_bound_ip_address', 'upper_bound_ip_address']
data_all = data_all.drop(cols_dropped, axis=1)

#bin countries
data_all.loc[data_all['country_count']<200, 'country'] = 'Small' #50 countries with counts > 200.

#reduce skewness of numeric features
numeric_features = data_all.select_dtypes(include='number').columns.tolist()
numeric_features.remove('class')
skewness = data_all[numeric_features].apply(lambda x: skew(x.dropna())).sort_values()
skew_features = skewness[abs(skewness)>0.5].index
for feature in skew_features:
    data_all[feature] = np.log1p(data_all[feature]) #log transform to reduce feature skewness

#feature encoding
le = preprocessing.LabelEncoder().fit(data_all['device_id'])
data_all['device_id'] = le.transform(data_all['device_id'])

#One hot encoding
data_all = pd.get_dummies(data_all)

#columns re-arrangement
cols = data_all.columns.tolist()
cols = cols[0:5] + cols[6:-1] + cols[5:6]
data_all = data_all[cols]

# train_test data split
test_size = 0.3
X = data_all.iloc[:, :-1]
y = data_all.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

#modeling
#RandomForest
metrics_rf = []
rf = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini', max_features='log2', min_samples_split=3,
                            random_state=0, n_jobs=-1)
# rf = RandomForestClassifier(random_state=0, n_jobs=-1)
# params = {'n_estimators': [100,1000], 'max_depth': [3,5], 'max_features': ['log2', 'sqrt'], 'criterion': ['gini', 'entropy'],
#           'min_samples_split': [2,3]}
# rf = GridSearchCV(rf, params, cv=3, verbose=1, n_jobs=-1) #parameters tuning
rf.fit(x_train, y_train)
y_hat_rf = rf.predict(x_test)
y_hat_rf_proba = rf.predict_proba(x_test)[:,1]
tpr_rf, fpr_rf, threshold = roc_curve(y_test, y_hat_rf_proba)

metrics_rf.append(accuracy_score(y_test, y_hat_rf))
metrics_rf.append(precision_score(y_test, y_hat_rf))
metrics_rf.append(recall_score(y_test, y_hat_rf))
metrics_rf.append(f1_score(y_test, y_hat_rf))
metrics_rf.append(auc(tpr_rf, fpr_rf))
metrics_rf.append(confusion_matrix(y_test, y_hat_rf)) #tn, fp, fn, tp

#XGBoost
metrics_xgb = []
xgb = xgb.XGBClassifier(max_depth=3,learning_rate=0.1, n_estimators=100, objective='binary:logistic', booster='gbtree',
                       reg_alpha=0, reg_lambda=1, random_state=0, n_jobs=4)
# xgb = xgb.XGBClassifier()
# params = {'max_depth':[3,5], 'learning_rate':[0.1,0.2], 'n_estimators':[100,1000,5000], 'objective':['binary:logistic'],
#          'booster': ['gbtree'], 'reg_alpha': [0.1,0.5,1], 'reg_lambda':[0.1,0.5,1]}
# xgb = GridSearchCV(xgb, params, cv=3, verbose=1, n_jobs=-1) #parameters tuning
xgb.fit(x_train, y_train)
y_hat_xgb = xgb.predict(x_test)
y_hat_xgb_proba = xgb.predict_proba(x_test)[:,1]
tpr_xgb, fpr_xgb, threshold = roc_curve(y_test, y_hat_xgb_proba)

metrics_xgb.append(accuracy_score(y_test, y_hat_xgb))
metrics_xgb.append(precision_score(y_test, y_hat_xgb))
metrics_xgb.append(recall_score(y_test, y_hat_xgb))
metrics_xgb.append(f1_score(y_test, y_hat_xgb))
metrics_xgb.append(auc(tpr_xgb, fpr_xgb))
metrics_xgb.append(confusion_matrix(y_test, y_hat_xgb))

metrics_terminology = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix']
metrics = pd.DataFrame({'Metrics':metrics_terminology, 'RandomForest': metrics_rf, 'XGBoost': metrics_xgb})

feature_importance = pd.DataFrame({'Features': x_train.columns,'RandomForest': rf.feature_importances_,
                                   'XGBoost': xgb.feature_importances_}).sort_values('XGBoost', ascending=False).head(15)
plt.figure(figsize=(8,8))
plt.plot([0,1],[0,1], c='k', linestyle='--')
plt.plot(tpr_rf, fpr_rf, label='RandomForest AUC: %.4f' % auc(tpr_rf, fpr_rf), c='g', linestyle='-')
plt.plot(tpr_xgb, fpr_xgb, label='XGBoost AUC: %.4f' % auc(tpr_xgb, fpr_xgb), c='b', linestyle='-')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Fraud detection - Receiver Operating Characterisics')
plt.legend(loc='lower right')
plt.plot()

print metrics
print feature_importance