from random import Random

import pandas as pd
from pandas.core.common import random_state

from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("report.html")

target = "Outcome"
x = data.drop("Outcome", axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Dựa vào correlation matrix để xét tính phi tuyến của mô hình, từ đó chọn dạng mô hình phù hợp (tuyến tính, phi tuyến)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model = LogisticRegression(random_state=100)
# model.fit(x_train, y_train)
params = {
    "n_estimators" : [50, 100, 200],
    "criterion" : ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5]
}
model = GridSearchCV(RandomForestClassifier(random_state = 100), param_grid=params, scoring = "recall", cv=6, verbose=2)
model.fit(x_train, y_train)
print(f"Best score: {model.best_score_}")
print(f"Best params: {model.best_params_}")
y_predict = model.predict(x_test)

# correct = 0
# for i,j in zip(y_predict, y_test):
#     print(f"Predicted value: {i}. Actual value: {j}")
#     if i == j:
#         correct+=1
#
# print(correct*100/len(y_predict))

# print(f"Accuracy score: {accuracy_score(y_test, y_predict)}")
# print(f"F1 score: {f1_score(y_test, y_predict)}")
print(classification_report(y_test, y_predict))
# Tùy bài toán mà ta tập trung vào thông số precision, recall hoặc f1 chứ không rập khuôn