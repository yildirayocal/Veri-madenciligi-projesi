import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Kurgusal kitap satış mağazası veri kümesi oluştur
book_data = dict(KitapAdi=['İstanbul Hatırası', 'Kürk Mantolu Madonna', 'Beyaz Kale', 'Suskunlar', 'Zaman Makinesi'],
                 Yazar=['Ahmet Umit', 'Sabahattin Ali', 'Orhan Pamuk', 'İhsan Oktay Anar', 'H.G. Wells'], Fiyat=[30, 25, 40, 20, 35],
                 SatışMiktarı=[100, 80, 120, 70, 90])

df_books = pd.DataFrame(book_data)

# Basit bir lineer regression modeli oluştur
X_books = df_books[['Fiyat']]
y_books = df_books['SatışMiktarı']

X_train_books, X_test_books, y_train_books, y_test_books = train_test_split(X_books, y_books, test_size=0.2, random_state=42)

model_books_lr = LinearRegression()
model_books_rf = RandomForestClassifier(random_state=42)

# Lineer Regresyon ve Random Forest modellerini birleştir
ensemble_model_books = VotingClassifier(estimators=[('lr', model_books_lr), ('rf', model_books_rf)], voting='soft')
ensemble_model_books.fit(X_train_books, y_train_books)

# Model performansını değerlendir
y_pred_books_ensemble = ensemble_model_books.predict(X_test_books)
mse_books_ensemble = mean_squared_error(y_test_books, y_pred_books_ensemble)
print("\nKitap Satışları İçin Ensemble Model Mean Squared Error:", mse_books_ensemble)

# Kurgusal müşteri veri kümesi oluştur
customer_data = {
    'MusteriID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Yas': [25, 30, 22, 35, 28, 40, 45, 32, 29, 38],
    'Cinsiyet': ['Erkek', 'Kadın', 'Erkek', 'Erkek', 'Kadın', 'Erkek', 'Kadın', 'Kadın', 'Erkek', 'Kadın'],
    'AylıkHarcama': [200, 150, 100, 250, 180, 300, 350, 220, 170, 280],
    'KampanyaKatilim': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df_customers = pd.DataFrame(customer_data)

# Kategorik değişkenleri sayısal değerlere dönüştür
label_encoder = LabelEncoder()
df_customers['Cinsiyet'] = label_encoder.fit_transform(df_customers['Cinsiyet'])

# Model için özellik ve hedef değişkenleri belirle
X_customers = df_customers[['Yas', 'Cinsiyet', 'AylıkHarcama']]
y_customers = df_customers['KampanyaKatilim']

# Veriyi eğitim ve test setlerine ayır
X_train_customers, X_test_customers, y_train_customers, y_test_customers = train_test_split(
    X_customers, y_customers, test_size=0.2, random_state=42
)

# Sınıflandırma modeli oluştur
model_customers_rf = RandomForestClassifier(random_state=42)

# Random Forest modelini kullanarak müşteri ilişkisi analizi
model_customers_rf.fit(X_train_customers, y_train_customers)

# Model performansını değerlendir
y_pred_customers_rf = model_customers_rf.predict(X_test_customers)
accuracy_customers_rf = accuracy_score(y_test_customers, y_pred_customers_rf)

print("\nMüşteri İlişkisi Analizi İçin Random Forest Accuracy:", accuracy_customers_rf)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_customers, y_pred_customers_rf))
print("\nClassification Report:")
print(classification_report(y_test_customers, y_pred_customers_rf))
