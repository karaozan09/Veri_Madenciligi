# Gerekli kütüphanelerin yüklenmesi
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Iris veri setini yükleme
iris = load_iris()
X = iris.data  # Özellikler
y = iris.target  # Hedef değişken

# Veri setini eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini oluşturma
knn = KNeighborsClassifier(n_neighbors=3)  # K=3
knn.fit(X_train, y_train)  # Modeli eğitme

# Test veri seti üzerinde tahmin yapma
y_pred = knn.predict(X_test)

# Modelin doğruluğunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluk değeri:", accuracy)


from sklearn.metrics import confusion_matrix

# Karmaşıklık matrisini hesaplama
cm = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrisi:\n", cm)

# Doğruluk değerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluk değeri:", accuracy)