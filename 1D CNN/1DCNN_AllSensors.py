import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Wczytanie danych
df = pd.read_csv('emg.csv', header=None)
X = df.iloc[:, :-1].values

# Etykiety przemapowane od 0
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, -1].values)

# Normalizacja
scaler = StandardScaler()
X = scaler.fit_transform(X)

# CNN oczekuje danych 3D: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model CNN
model = models.Sequential([
    Input(shape=(X.shape[1], 1)),  # <-- zamiast input_shape w Conv1D
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trening

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,               # Zatrzymaj po 10 epokach bez poprawy
    restore_best_weights=True # Przywróć najlepszy model
)

checkpoint = ModelCheckpoint(
    'best_model.h5',           # Nazwa pliku do zapisu
    monitor='val_accuracy',    # Monitorujemy dokładność walidacyjną
    save_best_only=True,       # Zapisz tylko najlepszy
    mode='max',                # Chcemy maksymalizować val_accuracy
    verbose=1                  # Wypisz info przy zapisie
)


#history = model.fit(X_train, y_train, epochs, validation_data=(X_test, y_test), verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=500,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)



# Wyniki
print("Dokładność:", model.evaluate(X_test, y_test)[1])

# Wykres dokładności
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('CNN Dokładność')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.show()

# --- Ranking elektrod ---

# Wagi pierwszej warstwy Conv1D
conv1_weights = model.layers[0].get_weights()[0]  # shape: (kernel_size, input_channels, filters)

# Przekształć wagi tak, by były po "wejściowych cechach"
# Najpierw zredukuj wymiar filtra (suma/średnia po kernel_size i kanałach)
weights_per_input = np.mean(np.abs(conv1_weights), axis=(0, 2))  # shape: (1,) -> powielimy zaraz

# Rozciągnij do 80 cech, zgodnie z układem danych
feature_weights = np.zeros(80)

# Przesuwające się filtry po 80 cechach (z kernel_size=3)
# Średnia z wag, które obejmują daną cechę
kernel_size = conv1_weights.shape[0]
n_filters = conv1_weights.shape[2]

# Sumujemy wkład każdej cechy (z uwzględnieniem przesuwania)
for i in range(80 - kernel_size + 1):
    for k in range(kernel_size):
        feature_weights[i + k] += np.mean(np.abs(conv1_weights[k, 0, :]))

# Uśrednij wkład każdej cechy
feature_weights /= (80 - kernel_size + 1)

# Teraz 80 cech podziel na 8 czujników (po 10 cech)
electrode_importance = []
for i in range(8):
    weights = feature_weights[i*10:(i+1)*10]
    electrode_importance.append(np.sum(weights))

# Posortuj i narysuj
sorted_electrode_idx = np.argsort(electrode_importance)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(8), np.array(electrode_importance)[sorted_electrode_idx])
plt.xticks(range(8), sorted_electrode_idx + 1)
plt.xlabel("Numer elektrody")
plt.ylabel("Suma istotności cech")
plt.title("Ranking ważności elektrod w modelu CNN")
plt.grid(True)
plt.tight_layout()
plt.show()
