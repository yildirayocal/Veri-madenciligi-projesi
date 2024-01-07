import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Kurgusal kitap satış mağazası veri kümesi oluştur
data = {
    'KitapAdi': ['İstanbul Hatırası', 'Kürk Mantolu Madonna', 'Beyaz Kale', 'Suskunlar', 'Zaman Makinesi'],
    'Yazar': ['Ahmet Umit', 'Sabahattin Ali', 'Orhan Pamuk', 'İhsan Oktay Anar', 'H.G. Wells'],
    'Fiyat': [30, 25, 40, 20, 35],
    'SatışMiktarı': [100, 80, 120, 70, 90]
}

df = pd.DataFrame(data)

# Veri setini incele
print("Veri Seti:")
print(df)

# Basit bir lineer regresyon modeli oluştur
X = df[['Fiyat']]
y = df['SatışMiktarı']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Model performansını değerlendir
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Modelin grafiğini çiz
plt.scatter(X, y, color='blue', label='Gerçek Veriler')
plt.plot(X, model.predict(X), color='red', linewidth=3, label='Regresyon Modeli')
plt.title('Kitap Fiyatı ve Satış Miktarı İlişkisi')
plt.xlabel('Kitap Fiyatı')
plt.ylabel('Satış Miktarı')
plt.legend()
plt.show()
