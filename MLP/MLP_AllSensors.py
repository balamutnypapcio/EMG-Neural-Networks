import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# Wczytanie danych
try:
    df = pd.read_csv('emg.csv', header=None)
except FileNotFoundError:
    raise Exception("Plik emg.csv nie został znaleziony!")

X = df.iloc[:, :-1].values

# Etykiety przemapowane od 0
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, -1].values)


# Informacje o etykietach
unique_classes = np.unique(y)
num_classes = len(unique_classes)
print("Unikalne etykiety:", unique_classes)
print("Liczba klas:", num_classes)

# Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział na zbiór treningowy/testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Budowa modelu MLP
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Trenowanie modelu

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=500,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# Ewaluacja
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Dokładność na zbiorze testowym:", round(acc * 100, 2), "%")

# --- Wykres 1: Dokładność treningu
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu MLP')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mlp_accuracy_plot.png')
plt.show()

# --- Wykres 2: "Ważność" cech przez wagi wejściowe (proxy)
input_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(input_weights), axis=1)

top_features = np.argsort(feature_importance)[-10:][::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(10), feature_importance[top_features])
plt.xticks(range(10), top_features)
plt.xlabel("Numer cechy")
plt.ylabel("Średnia wartość absolutna wag")
plt.title("Top 10 najważniejszych cech wg pierwszej warstwy")
plt.tight_layout()
plt.savefig('mlp_feature_importance.png')
plt.show()


electrode_importance = []

# Dla każdej elektrody (grupy 10 cech)
for i in range(8):
    # Wagi dla cech powiązanych z tą elektrodą
    electrode_weights = input_weights[:, i*10:(i+1)*10]
    # Średnia wartość bezwzględna wag dla tych cech
    importance = np.mean(np.abs(electrode_weights), axis=1)
    # Oblicz średnią wagę dla całej elektrody (średnia po wszystkich cechach tej elektrody)
    electrode_importance.append(np.sum(importance))

# Posortuj elektrody po ważności (od największej do najmniejszej)
sorted_electrode_idx = np.argsort(electrode_importance)[::-1]

# Wykres 3: Ranking ważności elektrod
plt.figure(figsize=(12, 6))
plt.bar(range(8), np.array(electrode_importance)[sorted_electrode_idx])
plt.xticks(range(8), sorted_electrode_idx + 1)  # Numer elektrody (1-8)
plt.xlabel("Numer elektrody")
plt.ylabel("Całkowita waga (średnia waga cech)")
plt.title("Ranking elektrod na podstawie ich wpływu na klasyfikację gestu")
plt.tight_layout()
plt.savefig('electrode_importance.png')
plt.show()