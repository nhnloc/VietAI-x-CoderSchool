import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("dataset_2.csv")
target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

# imputer = SimpleImputer()
# x_train[["reading score", "writing score"]] = imputer.fit_transform(x_train[["reading score", "writing score"]])
#
# scaler = StandardScaler()
# x_train[["reading score", "writing score"]] = scaler.fit_transform(x_train[["reading score", "writing score"]])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False)),
])
education_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gender_order = x_train["gender"].unique()
lunch_order = x_train["lunch"].unique()
test_order = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_order, gender_order, lunch_order, test_order])),
])

pre_processor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),

])

model = Pipeline(steps=[
    ("pre_processor", pre_processor),
    ("regressor", LinearRegression())
])

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(f"Mean absolute error: {mean_absolute_error(y_test, y_predict)}")
print(f"Mean square error: {mean_squared_error(y_test, y_predict)}")


# processed_data = num_transformer.fit_transform(x_train[["reading score", "writing score"]])
# processed_data = ord_transformer.fit_transform(x_train[["parental level of education"]])
# for i, j in zip(x_train[["parental level of education"]].values, processed_data):
#     print(f"Before: {i}, After: {j}")
# print(data["lunch"].unique())

# Mean absolute error: 4.189583645430788
# Mean square error: 28.000700798773877