import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Kurgusal kitap satış mağazası veri kümesi oluştur
book_data = {
    'KitapAdi': ['İstanbul Hatırası', 'Kürk Mantolu Madonna', 'Beyaz Kale', 'Suskunlar', 'Zaman Makinesi' , 'Kan Bağı', 'Silent Scream', 'Kızıl Ejder', 'Mustafa Kemal Atatürk', 'Benim Adım Feridun'],
    'Yazar': ['Ahmet Umit', 'Sabahattin Ali', 'Orhan Pamuk', 'İhsan Oktay Anar', 'H.G. Wells', 'Tess Gerritsen', 'Angela Marsons', 'Thomas Harris', 'Andrew Mango', 'Tuna Kiremitçi'],
    'Fiyat': [30, 25, 40, 20, 35, 30, 25, 40, 20, 35],
    'SatışMiktarı': [100, 80, 120, 70, 90, 100, 80, 120, 70, 90]
}

df_books = pd.DataFrame(book_data)

# Veri setini incele
print("Kitap Satış Veri Seti:")
print(df_books)

# Basit bir lineer regresyon modeli oluştur
X_books = df_books[['Fiyat']]
y_books = df_books['SatışMiktarı']

X_train_books, X_test_books, y_train_books, y_test_books = train_test_split(X_books, y_books, test_size=0.2, random_state=42)

model_books = LinearRegression()
model_books.fit(X_train_books, y_train_books)

# Model performansını değerlendir
y_pred_books = model_books.predict(X_test_books)
mse_books = mean_squared_error(y_test_books, y_pred_books)
print("\nKitap Satışları İçin Mean Squared Error:", mse_books)

# Modelin grafiğini çiz
plt.scatter(X_books, y_books, color='blue', label='Gerçek Veriler')
plt.plot(X_books, model_books.predict(X_books), color='red', linewidth=3, label='Regresyon Modeli')
plt.title('Kitap Fiyatı ve Satış Miktarı İlişkisi')
plt.xlabel('Kitap Fiyatı')
plt.ylabel('Satış Miktarı')
plt.legend()
plt.show()

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

# Veri setini incele
print("\nMüşteri Veri Seti:")
print(df_customers)

# Model için özellik ve hedef değişkenleri belirle
X_customers = df_customers[['Yas', 'Cinsiyet', 'AylıkHarcama']]
y_customers = df_customers['KampanyaKatilim']

# Veriyi eğitim ve test setlerine ayır
X_train_customers, X_test_customers, y_train_customers, y_test_customers = train_test_split(
    X_customers, y_customers, test_size=0.2, random_state=42
)

# K-en yakın komşu (KNN) sınıflandırma modeli oluştur
knn_model_customers = KNeighborsClassifier(n_neighbors=3)
knn_model_customers.fit(X_train_customers, y_train_customers)

# Model performansını değerlendir
y_pred_customers = knn_model_customers.predict(X_test_customers)
accuracy_customers = accuracy_score(y_test_customers, y_pred_customers)

print("\nMüşteri İlişkisi Analizi İçin Accuracy:", accuracy_customers)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_customers, y_pred_customers))
print("\nClassification Report:")
print(classification_report(y_test_customers, y_pred_customers))
