import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("./dataset/news.csv")

labels = data.label

text = data.text    

training_text, validation_text, training_labels, validation_labels = train_test_split(text,labels,train_size=0.8,random_state=10)

vectorizer = TfidfVectorizer(stop_words="english",max_df=0.7)
train_vectorized = vectorizer.fit_transform(training_text)
test_vectorized = vectorizer.transform(validation_text)

model = PassiveAggressiveClassifier(max_iter=100)
model.fit(train_vectorized,training_labels)
y_pred = model.predict(test_vectorized)
score = accuracy_score(validation_labels,y_pred)
print(f"Accuracy:{round(score*100,2)}%")
cfmat = confusion_matrix(validation_labels,y_pred,labels=['FAKE','REAL'])
print(cfmat)

# Predicting Sample
sample_text = np.array(["Google was destroyed by amazon last year"])
sample_vect = vectorizer.transform(sample_text)
sample_pred = model.predict(sample_vect)
print(sample_pred)

# Predicing Sample 2
sample_text2 = np.array(["Trump serves as the president of USA"])
sample_vect2 = vectorizer.transform(sample_text2)
sample_pred2 = model.predict(sample_vect2)
print(sample_pred2)