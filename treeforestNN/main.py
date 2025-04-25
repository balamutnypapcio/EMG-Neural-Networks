import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tkinter
import matplotlib
matplotlib.use('TkAgg')  # Możesz spróbować także 'Qt5Agg'
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
df = pd.read_csv('emg.csv', header=None)  # Zmień ścieżkę na odpowiednią

# Podział na cechy (X) i etykiety (y)
X = df.iloc[:, :80].values  # Pierwsze 80 kolumn to cechy
y = df.iloc[:, 80].values   # Ostatnia kolumna (81) to etykieta (klasyfikacja)

# Podział na dane treningowe i testowe (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Ocena
print("Dokładność: ", accuracy_score(y_test, y_pred))

# --- Wykres 1: Zmienność dokładności w czasie (dla Random Forest, np. po każdej iteracji estymatora)
# Możemy uzyskać dokładność na zbiorze treningowym i testowym po każdej iteracji estymatora
train_accuracy = []
test_accuracy = []

for i in range(1, 101):
    model = RandomForestClassifier(n_estimators=i)
    model.fit(X_train, y_train)

    # Ocena dokładności na zbiorze treningowym i testowym
    train_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_accuracy, label="Treningowa dokładność", color="blue")
plt.plot(range(1, 101), test_accuracy, label="Testowa dokładność", color="green")
plt.xlabel("Liczba drzew w modelu")
plt.ylabel("Dokładność")
plt.title("Ewolucja dokładności modelu Random Forest")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('output_plot1.png')  # Zapisuje wykres do pliku PNG

# --- Wykres 2: Feature Importance (ważność cech)
importances = model.feature_importances_
indices = importances.argsort()[::-1]

# Tworzenie wykresu dla 10 najważniejszych cech
plt.figure(figsize=(12, 6))
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), indices[:10])
plt.xlabel("Numer cechy")
plt.ylabel("Ważność")
plt.title("10 najważniejszych cech w Random Forest")
plt.show()
plt.savefig('output_plot2.png')  # Zapisuje wykres do pliku PNG