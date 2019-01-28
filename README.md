# Fraud detection: Exploring feature extraction for predicting fraudulent activity of the first transaction of new users.

This repo explores feature extraction from raw dataset to build machine learning models to detect fraudulent transactions occurring in an e-commerce website. The original dataset contains both bioinfos of new users (device id, ip address, source, browser, age, country and sex) and their activities (signup time, purchase time, purchase value). Exploratory data analysis (EDA) suggests that flash transaction observed by counting time difference between purchase_time and signup_time is a strong indication of a fraud. Thus, here we demonstrate two models: one with full dataset and the other modeled without flash transaction observations, where the latter was then re-combined with flash transaction observations for comparing the metrics of these two models. The user experience has also been recommended to minimizing business loss but at the same time minimizing obstruction of web-visiting traffic. 

#Background
  
  Transaction in e-commerce websites has high risk of users performing fraudulent activities such as doing money laundry, using stolen identity and credit card, etc. due to unscreened and diversified background of users. The rise in artificial intelligent (machine learning in specific) enables to detect these fraudulent activities with accuracy and real-time. However, the activities and background of a new user visiting a website are usually not directly learnable by traditional machine learning algorithms. Feature extraction from activity records and feature transformation from user's background are a necessity as well as a prerequisite.
    
   In this repo, we have a dataset containing a website selling clothes and the first transaction activities of new users. The users' activity and background are collected when they visited and performed activities at the website. Here the purpose is to build a model to predict (in real time) whether a new user is performing fraudulent activity. Also, based on model outputs, what kind of users' experience could be recommended to the seller.

#Feature extraction

  The raw data has the columns of user id, signup time, purchase time, device id, source, browser, sex, age, ip address, class, and country. Features including user id, source, browser, sex, age, and country can be classified as identity characteristics, which can be directly transformed for machine learning modeling. However, features including signup time, purchase time, device id, and ip address are activity-based that do not carry machine-recoganizable patterns as themselves. Rather, we could extract much more valuable characteristics from these than identity characteristics for machine learning modeling, since it is the activity differentiating one user from the other that could teach and train a model. So what active features can be extracted?
  
  |signup_time|purchase_time|purchase_value|device_id|browser|age|ip_address|country|
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |1.000|0.9971|0.0008|0.9129|0.00003|0.00038|0.9497|0.00119|  
  
Table 1. Data's uniqueness in each column from the original dataset. 
  
  1). The original signup time and purchase time are random, continuous and 100% unique, but the time difference ("time_diff") between signup time and purchase time may show characteristics of fraudulent activity, such as flash transaction suggesting autonomous trading, constant or perodic time difference may also suggest fraudulent autonomous trading, etc.
  
  2). The ip address itself again does not carry characteristics for modeling (~95% data uniqueness). However, if many users use the same ip address, the probability of conducting fraudulent activity could be high. Thus, feature "ip_users" (how many unique users use the same ip address?) will catch such activities.
  
  3). Similarly, that many users use the same device for transaction suggests high possibility of fraudulent activity. Thus, feature of "device_id_unique_users" (how many unique users use the same device_id for transaction?) could be a nice detector.
  
  4). From signup time and purchase time, we can also create features of what day / week transactions are more frequently occurred. If there were traceable patterns of transactions occuring, it again shows more possibility of fraudulent activities. Thus, we create features of "week_of_the_day" and "week_of_the_year" for both signup time and purchase time.
  
  5). Besides, since device_id can be shared by many unique users, "total_purchase" and/or "average_purchase" for each device_id may imply whether activities are fraudulent or not.
  
  6). Fraudulent activities might occur more frequently in a particular region (country here). So, having a feature showing number of users ('country_count') from the same country might be useful (or redundant). Also, since there are over 200 countries, we may use 'country_count' to bin those countries showing relatively less entries, say, less than 200, as a single group.
  
  7). 'purchase time' is not unique, which could be coincident due to high transaction traffic or could result from automatic transaction. Thus, here we create a feature 'purchase_times' by grouping-by the same purchase time.
  
#EDA

![time_diff](https://user-images.githubusercontent.com/34787111/51810209-4fefe180-225b-11e9-818a-9aee7a080cbc.png)

Figure 1. Statistical distributions of time_diff among two classes from origianl (left) and non-flash-transaction (right) datasets.

 EDA was performed comprehensively in order to have a first understanding on the data and provides guidance for feature selection (please refer to Fraud_EDA.ipynb for more details). Here we briefly discuss three of the most informatives. The time_diff distribution in original data (Figure 1, left) clearly indicates that there is a unique distribution in short time_diff for fraud only, which suggest us to separate that part of data and then replot it (as shown in Figure 1, right). When went back to check the original data for that short time_diff part, they all show time_diff 1s with class 1 (fraud). Surpringly, there is almost identical between two classes after eliminating the 1s transaction time difference.
 
 ![device_id_unique_users](https://user-images.githubusercontent.com/34787111/51810212-52ead200-225b-11e9-854c-8f8073f9cfc0.png)

Figure 2. Statistical distributions of device_id_unique_users among two classes from origianl (left) and non-flash-transaction (right) datasets.

   Similaryly, statistical plot of device_id_unique_users also show a distinct distribution region for fraud class, which has been completely removed after eliminating the 1s transaction observations (Figure 2). The ip_users shows strong linearity with device_id_unique_users when ip_users>1, which has been dropped for model for eliminating overfit from these two features.
   
![total_purchase](https://user-images.githubusercontent.com/34787111/51810216-554d2c00-225b-11e9-8378-5fc3446550b2.png)

Figure 3. Statistical distributions of total_purchase among two classes from origianl (left) and non-flash-transaction (center) datasets and avg_purchase (right).

   Statistical plot of total_purchase is more spreading, but again it is largely contributed from the 1s transaction observation. By comparing to avg_purchase, it does reveal that fraud tends to.pile up more purchase values by carrying out more frequent transactions (Figure 3).

#Model

  Data were further processed to reduce skewness of numeric features, scale down large values of numeric features, encode categorical features, and drop redundant / correlated features (see details in Fraud_flash.py). Then the data was splitted into train, val, and test sets. In order to have fair comparison, the data for non-flash-transaction model was splitted with exactly the same observations for train, val and test (by inheriting index of each splitted category from original splits).
  
|Metrics|original_val|original_test|non-flash_val|non-flash_test|combined_val|combined_test|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|auc|0.854|0.848|0.671|0.668|0.854|0.847|

Table 2. Comparison of auc for models using original dataset, non-flash-transaction dataset, and re-combined non-flash-transaction and flash-transaction with both val and test.  

  Due to imbalanced nature of observations, we chose threshold-independent receiver operating characteristics (roc) curve to evaluate model performance. As shown in Figure 4 and Table 2, the xgboost model results in good performance and excellent generalization capability given that val and test metrics results are well consistent. The non-flash-transaction model performs worse than that of the original, suggesting that there are no strong feature(s) favoriting fraud in non-flash-transaction dataset, which is consistent with EDA. However, once we re-combine the prediction from non-flash-transaction model and result (all given 1) from flash-transaction, the re-combined model shows exactly the same performance as the original. This infers that the observations from flash-transaction do not help model better 'generalize' those non-flash-transaction observations, likely partly due to the data nature (the first transactions of the new users). From roc, we can extract precision and recall by setting a threshold that is dependent on business model. If the bussiness prefer to minimizing false negatives, default threshold (0.5) works well (like the predictions from the two models). However, if the bussiness prefer to maximizing true positives, we can decrease threshold to predict more positives, though at the expense of predicting more false positives.

![roc](https://user-images.githubusercontent.com/34787111/51809237-97726f80-2253-11e9-8978-a62fca8fb277.png)
  Figure 4. ROC curves for fraud detection from XGBoost model (left: validation data, right: test data).
  
  To evaluate the importance / usefulness of features, we extract feature importances from models. As shown in Table 3, the extracted features are almost listed as the top 10 most important features in both models. Interestingly, both models show that 'time_diff' and 'device_id_unique_users' are the two most important features, though for non-flash-transaction dataset, the statistical distribution in 'time_diff' and 'device_id_unique_users' are almost identical between fraud and normal.

|Rank|Original|non-flash-transaction|
|:---:|:---:|:---:|
|1|time_diff|time_diff|
|2|device_id_unique_users|device_id_unique_users|
|3|total_purchase|age|
|4|age|total_purchase|
|5|source_Direct|country_count|
|6|country_count|source_Direct|
|7|country_Belgium|country_Belgium|
|8|DOTW_purchase_Saturday|WOTY_purchase_3|
|9|WOTY_purchase_43|browser_Safari|
|10|WOTY_purchase_14|DOTW_purchase_Wednesday|

Table 3. The top 10 most important features based on models using original dataset and non-flash-transaction dataset.

#Summary and business suggestion

  We demonstrated solution for detecting frauds of the first transaction of new users in e-commerce business. By EDA and modeling, it is reveled that flash transaction is a strong indication of fraudulent activity. There is still room for further improvement in fraud detection in non-flash-transaction observation, which can be collecting more observations or collecting more infos from each observation, or both.
  Given that we have a model enabling to predict the probability of committing a fraudulent activity, we may create different user experience to minimize fraudulent loss but maximize experice for normal users:
  1). If a user fraudulent probability is less than a threshold X (say 0.1), the user will have normal experience;
  2). If a user fraudulent probability is larger than X but lower than a highly-suspicious threshold Y, the user can be directed to a page asking for additional verification via SMS or socical network account like Facebook;
  3). If a user fraudulent probability is larger than Y (say 0.5), then this user should be prohibited or at least put on hold to check his / her login data for further decision.
