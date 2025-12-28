import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

df = pd.read_csv("reviews.csv")
df['clean_review'] = df['review'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully")
