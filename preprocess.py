import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Explicit NLTK data path (Render-safe)
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
