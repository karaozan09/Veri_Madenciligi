import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Veri kümesini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veri kümesini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bayes sınıflandırıcısını oluştur
naive_bayes = GaussianNB()

# Modeli eğit
naive_bayes.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = naive_bayes.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)
