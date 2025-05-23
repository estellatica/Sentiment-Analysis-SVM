# 1. Import library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# 2. Load dataset (contoh: dataset sederhana dari GitHub atau dataset lokal)
# Untuk demo, kita pakai data buatan
data = {
    'text': [
        "I love this movie, it's amazing!",
        "This film is terrible, I hate it.",
        "Absolutely fantastic! Would watch again.",
        "Worst movie ever. Not worth it.",
        "I enjoyed every second of it.",
        "I can't stand this film.",
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)

# 3. Preprocessing teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# 4. TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Contoh prediksi
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = svm_model.predict(vectorized)
    return prediction[0]

# Contoh penggunaan
example = "I really love this product!"
print(f"Input: {example}")
print(f"Predicted Sentiment: {predict_sentiment(example)}")
    