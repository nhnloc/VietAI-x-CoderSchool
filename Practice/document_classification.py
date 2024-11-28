import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def filter_location(location):
    result = location.split(",")
    if len(result) > 1:
        return result[1][1:]
    else:
        return location

data = pd.read_excel("job_dataset.ods", engine="odf", dtype="str")
data = data.dropna(axis=0)
data = data.drop(data[data['career_level'] == 'specialist'].index, axis=0)
data = data.drop(data[data['career_level'] == 'managing_director_small_medium_company'].index, axis=0)
data["location"] = data["location"].apply(filter_location)
# print(data.info())

target = "career_level"

# print(data[target].value_counts())

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# bereichsleiter = senior_manager_head_of_department

# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
# processed_data = vectorizer.fit_transform(x_train['description'])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# # print(data['title'][0])
# print(processed_data.shape)

# encoder = OneHotEncoder()
# processed_data = encoder.fit_transform(x_train[['location']])
# print(processed_data.shape)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "title"),
    ("location", OneHotEncoder(handle_unknown='ignore'), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1,2)), "description"),
    ("function", OneHotEncoder(handle_unknown='ignore'), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "industry"),
])

model = Pipeline(steps=[
    ("pre-processor", preprocessor),
    ("classify", RandomForestClassifier(random_state=100))
])

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(classification_report(y_test, y_predict))
