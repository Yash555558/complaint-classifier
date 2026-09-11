import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("complaints.csv")

def preprocess_text(text):
    return text

data["complaint_text"] = data["complaint_text"].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["complaint_text"])

y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = MultinomialNB()
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)

def predict(text):
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text])
    predicted_category = model1.predict(vectorized_text)[0]
    return predicted_category

def subpredict(category, tex):
    filtered_data = data[data["category"] == category]

    if filtered_data.empty:
        return "No subcategory found for this category."

    X1 = vectorizer.transform(filtered_data["complaint_text"])
    y1 = filtered_data["subcategory"]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    model2 = MultinomialNB()
    model2.fit(X1_train, y1_train)

    tex = preprocess_text(tex)
    vectorized_text = vectorizer.transform([tex])

    predicted_subcategory = model2.predict(vectorized_text)[0]
    return predicted_subcategory
i=0
while i==0 :
    print("Enter your problem: ")
    new_complaint = input()
    predicted_category = predict(new_complaint)
    predicted_subcategory = subpredict(predicted_category, new_complaint)
    print("Predicted Category: ", predicted_category)
    print("Predicted Sucbategory: ", predicted_subcategory)