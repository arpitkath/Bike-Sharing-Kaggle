import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, gradient_boosting, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, f1_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss
from math import log, sqrt
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv("train.csv", parse_dates=["datetime"])
test = pd.read_csv("test.csv", parse_dates=["datetime"])
target = train["count"]
plt.scatter(train["humidity"], train["count"])
plt.show()
train.drop(["count", "casual", "registered"], axis=1, inplace=True)
print(train.shape, test.shape)
date_time = pd.DataFrame(test["datetime"])
date_time = pd.to_datetime(pd.Series(date_time), format="%Y-%m-%d")
train = pd.concat([train, test], axis=0, ignore_index=True)

train["hour"] = pd.DatetimeIndex(train["datetime"]).hour
train["date"] = pd.DatetimeIndex(train["datetime"]).day
train["day_of_week"] = pd.DatetimeIndex(train["datetime"]).dayofweek
train["month"] = pd.DatetimeIndex(train["datetime"]).month
train["year"] = pd.DatetimeIndex(train["datetime"]).year
train["week"] = pd.DatetimeIndex(train["datetime"]).week
train["quarter"] = pd.DatetimeIndex(train["datetime"]).quarter
'''
train["is_day_time"] = 0
train["is_day_time"].loc[train["hour"] >= 8][train["hour"] <= 20] = 1
train["travel_time"] = 0
train["travel_time"][train["hour"] == 7][train["hour"] == 8][train["hour"] >= 17][train["hour"] <= 19] = 1
'''
train.drop("datetime", axis=1, inplace=True)
train["isHumid"] = 0
train["isHumid"][train["humidity"] > 77] = 1

train["windspeed"][train["windspeed"] == 0] = np.nan
train["windspeed"][train["windspeed"] > 25] = np.nan
train["windspeed"].fillna(train["windspeed"].mean(), inplace=True)

#print(train.describe())
season_dummy = pd.get_dummies(train["season"], prefix="season", prefix_sep="_")
train = train.join(season_dummy)

weather_dummy = pd.get_dummies(train["weather"], prefix="weather", prefix_sep="_")
train = train.join(weather_dummy)

year_dummy = pd.get_dummies(train["year"], prefix="year", prefix_sep="_")
train = train.join(year_dummy)
train.drop("year", axis=1, inplace=True)

quarter_dummy = pd.get_dummies(train["quarter"], prefix="Q")
train = train.join(quarter_dummy)
test = train[train["date"] > 19]
train = train[train["date"] <= 19]
train.to_csv("abc.csv")
'''
plt.scatter(train["quarter"], train["count"])
plt.show()
'''
test = train[train["date"] > 19]
train = train[train["date"] <= 19]
print(train.shape, test.shape)
#train_X, test_X, train_Y, test_Y = train_test_split(train, target)


dt = DecisionTreeRegressor()
dt = dt.fit(train, target)
cross = cross_val_score(dt, train, target, cv=10)
print(cross.mean())

from sklearn.ensemble import AdaBoostRegressor
ad = AdaBoostRegressor(n_estimators=1000)
ada = ad.fit(train, target)
cross = cross_val_score(ada, train, target, cv=10)
print(list(target))
print(list(ada.predict(train)))
print(cross.mean())

'''
rf = RandomForestRegressor(n_estimators=10)
rf = rf.fit(train, target)
cross = cross_val_score(rf, train, target, cv=10)
print(cross.mean())
print(list(train.keys()))
print((np.array(rf.feature_importances_)*100))
print(train.describe())
#err = target - pred_val

#print(pd.isnull(count1).count())
#print(count1)
#train['count1_'] = target2
#test['count1_'] = np.array(count1)

rf = BaggingRegressor(n_estimators=100)
rf = rf.fit(train, pd.Series(target))
cross = cross_val_score(rf, train, target, cv=10)
print(2,cross.mean())



rf = RandomForestRegressor(n_estimators=100)
rf = rf.fit(train, pd.Series(target))
cross = cross_val_score(rf, train, target, cv=10)
print(3,cross.mean())
print(list(train.keys()))
print(rf.feature_importances_)
#print(pd.isnull(test["count1_"]).count())
count = pd.Series(rf.predict(test))
'''
'''
rf = SVR(kernel='poly')
rf = rf.fit(train, pd.Series(target))
cross = cross_val_score(rf, train, target, cv=10)
print(4,cross.mean())
#print(pd.isnull(test["count1_"]).count())
count3 = pd.Series(rf.predict(test))


count = pd.Series((np.array(count))/4+(np.array(count1))/4+(np.array(count2))/4 + (np.array(count3))/4)
count[count <= 0] = 1


date_time = date_time.astype(str)
#date_time = np.array(date_time)
solution = pd.DataFrame(date_time)
solution = pd.concat([solution, count], axis=1)
solution.columns = ["datetime", "count"]
solution.to_csv("solution.csv", index=False, index_label=False, date_format="%Y-%m-%d")'''