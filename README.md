# 📊 Sentiment Analysis menggunakan Support Vector Machine (SVM)

## 📝 Deskripsi Proyek
Proyek ini merupakan implementasi dasar dari *Sentiment Analysis* menggunakan algoritma **Support Vector Machine (SVM)**. Sentiment analysis adalah proses untuk mengidentifikasi dan mengklasifikasikan opini atau sentimen dalam teks, seperti positif, negatif, atau netral.

SVM adalah salah satu algoritma machine learning yang populer untuk klasifikasi karena kemampuannya dalam menangani data berdimensi tinggi dan hasil klasifikasi yang akurat.

## 📚 Referensi Pembelajaran
Saya menggunakan referensi dari:
- [Medium - Sentiment Analysis using SVM](https://medium.com/scrapehero/sentiment-analysis-using-svm-338d418e3ff1) 

## 🛠️ Tools & Teknologi
- Python
- Scikit-learn
- Pandas
- Numpy
- Jupyter Notebook / Google Colab

## 🚀 Langkah-langkah Implementasi
1. Mengimpor dataset dari `{nama_dataset.csv}`.
2. Melakukan preprocessing teks: 
   - Lowercasing
   - Stopwords removal
   - Tokenization
   - Vectorization menggunakan TF-IDF
3. Membagi data menjadi data latih dan data uji.
4. Melatih model menggunakan SVM (`sklearn.svm.SVC`).
5. Melakukan prediksi dan evaluasi model.

## 📈 Hasil Evaluasi
Berikut ini adalah hasil evaluasi dari model yang saya jalankan:

- Akurasi: `{masukkan_angka_akurasi}`
- Precision: `{masukkan_angka_precision}`
- Recall: `{masukkan_angka_recall}`
- F1-score: `{masukkan_angka_f1}`

Contoh output prediksi:

```text
Input: "I really love this product!"
Predicted Sentiment: Positive
