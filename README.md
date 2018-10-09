# Fraud detection: Exploring feature extraction for predicting fraudulent activity of the first transaction of a new user.

This repo exemplifies feature extraction from raw dataset for building machine learning models to predict fraudulent transactions occurring in e-commerce websites. The original dataset contains all activities of a new user visiting a website, including sign-up time, purchase time, purchase value, ip address, device id, source, age and sex. Most of these activity records do not show patterns modelable by machine learning algorithms, which need either feature transformation or feature extraction for ML to learn and then predict. 

#Background
  
  Transaction in e-commerce websites has high risk of users performing fraudulent activities such as doing money laundry, using stolen identity and credit card, etc. due to unscreened and diversified background of users. The rise in artificial intelligent (machine learning in specific) enables to detect these fraudulent activities with accuracy and real-time. However, the activities and background of a new user visiting a website (called data) are usually not directly learnable by machine learning algorithms. Feature extraction from activity records and feature transformation from user's background are a necessity as well as a prerequisite for building a machine learning model.
    
   In this repo, we have a dataset containing a website selling clothes and the first transaction activities of new users. The users' activity and background are collected when they visited and performed activities at the website. Here the purpose is to build a machine learning model to predict (in real time) whether a new user is performing fraudulent activity. Also, based on model outputs, what kind of users' experience could be recommended to the website.

#Feature extraction

  The raw data has the columns of user id, signup time, purchase time, device id, source, browser, sex, age, ip address, class, and country. Features including user id, source, browser, sex, age, and country can be classified as identity characteristics, which can be directly transformed for machine learning modeling. However, features including signup time, purchase time, device id, and ip address are activity-based that do not carry machine-recoganizable patterns as themselves. Rather, we could extract much more valuable characteristics from these than identity characteristics for machine learning modeling, since it is the activity differentiating one user from the other that could teach and train a model. So what active features can be extracted?
  1). The original signup time and purchase time are random, continuous and 100% unique, but the time difference ("time_diff") between signup time and purchase time may show characteristics of fraudulent activity, such as flash transaction suggesting automative trading, a constant time difference may also suggest fraudulent automotive trading, etc.
  2). The ip address itself again does not carry characteristics for modeling (~95% data uniqueness). However, if many users use the same ip address, it may suggest fraudulent activity. Thus, we can extract feature of "ip_users" (how many unique users use the same ip address?).
  3). Similarly, that many users use the same device for transaction suggests high possibility of fraudulent activity. Thus, we can create feature of "device_id_unique_users" (how many unique users use the same device_id for transaction?).
  4). From signup time and purchase time, we can also create features of what day / week transactions are more frequently occurred. If there were traceable patterns of transactions occuring, it again shows more possibility of fraudulent activities. Thus, we could create features of "week_of_the_day" and "week_of_the_year" for both signup and purchase time.
  5). Besides, since device_id can be shared by many unique users, "total_purchase" and/or "average_purchase" for each device_id may also imply whether activities are fraudulent or not.
  6). In addition, fraudulent activities can occur in aggregation in a particular region (country here). So, having a feature showing number of users ('country_count') from the same country might be useful (or redundant).
  
  |signup_time|purchase_time|purchase_value|device_id|browser|age|ip_address|country|
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |1.0|0.997135|0.000807|0.912939|0.000033|0.000384|0.949706|0.001198|

  Table 1. Columns' uniqueness of the original dataset.

#Model

#Summary and business suggestion
