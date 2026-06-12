import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
train = pd.read_csv("hasil_train.csv")
test = pd.read_csv("hasil_test.csv")

# Fitur
X_train = train[["Contrast", "Correlation", "Energy", "Homogeneity"]]
y_train = train["Label"]

X_test = test[["Contrast", "Correlation", "Energy", "Homogeneity"]]
y_test = test["Label"]

# KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Training
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
