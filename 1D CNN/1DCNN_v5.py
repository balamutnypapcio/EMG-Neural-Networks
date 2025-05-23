import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import os

# Tworzenie folderu plots, jeśli nie istnieje
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Utworzono folder '{plots_dir}' na wykresy")

print("TensorFlow version:", tf.__version__)

# === Wczytanie danych ===
df1 = pd.read_csv('features_new.csv', sep=';', header=0)
df2 = pd.read_csv('features_new2.csv', sep=';', header=0)

# Sprawdź strukturę obu zbiorów danych
print("Kształt pierwszego zestawu danych:", df1.shape)
print("Kształt drugiego zestawu danych:", df2.shape)

# Połączenie zbiorów danych
df_combined = pd.concat([df1, df2], ignore_index=True)
print("\nKształt połączonego zbioru danych:", df_combined.shape)

# === Analiza struktury danych ===
# Sprawdź unikalne wartości kanałów, okien i gestów
channels = df_combined['Channel'].unique()
windows = df_combined['Window'].unique()
gestures = df_combined['GESTURE'].unique()

print(f"Liczba unikalnych kanałów: {len(channels)}")
print(f"Unikalne numery okien: {sorted(windows)}")
print(f"Liczba unikalnych gestów: {len(gestures)}")
print(f"Unikalne gesty: {sorted(gestures)}")


# === Funkcje do przetwarzania danych w sekwencje ===
def create_window_sequences(df, seq_length=3, overlap=0.5):
    """
    Tworzy sekwencje nakładających się okien dla każdego gestu i kanału.

    Args:
        df: DataFrame z danymi
        seq_length: Liczba kolejnych okien w sekwencji
        overlap: Współczynnik nakładania się okien w sekwencji

    Returns:
        X_seq: Tablica z sekwencjami cech
        y_seq: Tablica z etykietami dla sekwencji
        meta_seq: Tablica z metadanymi (gest, kanał, okno)
    """
    # Sortowanie danych według gestu, kanału i okna
    df_sorted = df.sort_values(['GESTURE', 'Channel', 'Window'])

    # Lista dostępnych cech (wszystkie kolumny oprócz Channel, Window i GESTURE)
    feature_cols = [col for col in df.columns if col not in ['Channel', 'Window', 'GESTURE']]

    sequences = []
    labels = []
    metadata = []  # (gest, kanał, [okna])

    # Dla każdego gestu
    for gesture in sorted(df['GESTURE'].unique()):
        # Dla każdego kanału
        for channel in sorted(df['Channel'].unique()):
            # Wybierz dane tylko dla obecnego gestu i kanału
            channel_data = df_sorted[(df_sorted['GESTURE'] == gesture) &
                                     (df_sorted['Channel'] == channel)]

            if len(channel_data) < seq_length:
                continue

            # Okna dla tego kanału i gestu
            windows_for_channel = sorted(channel_data['Window'].unique())

            # Ile okien przesuwamy się przy każdym kroku
            stride = max(1, int(seq_length * (1 - overlap)))

            # Tworzenie sekwencji
            for i in range(0, len(windows_for_channel) - seq_length + 1, stride):
                # Okna w tej sekwencji
                seq_windows = windows_for_channel[i:i + seq_length]

                # Sprawdź czy okna są kolejne
                if seq_windows[-1] - seq_windows[0] != seq_length - 1:
                    continue

                # Cechy dla tej sekwencji
                seq_features = []
                for window in seq_windows:
                    window_data = channel_data[channel_data['Window'] == window][feature_cols].values
                    if len(window_data) > 0:  # Upewnij się, że mamy dane dla tego okna
                        seq_features.append(window_data[0])

                if len(seq_features) == seq_length:
                    sequences.append(seq_features)
                    labels.append(gesture)
                    metadata.append((gesture, channel, seq_windows))

    return np.array(sequences), np.array(labels), np.array(metadata, dtype=object)


# === Przygotowanie danych w formie sekwencji ===
print("\n=== Tworzenie sekwencji nakładających się okien ===")
seq_length = 3  # Długość sekwencji (liczba kolejnych okien)
overlap = 0.5  # 50% nakładanie się okien w sekwencji

# Tworzenie sekwencji
X_seq, y_seq, meta_seq = create_window_sequences(df_combined, seq_length, overlap)

print(f"Kształt sekwencji danych: {X_seq.shape}")
print(f"Kształt etykiet: {y_seq.shape}")

# Kodowanie etykiet
le = LabelEncoder()
y_encoded = le.fit_transform(y_seq)
print(f"Liczba unikalnych klas po kodowaniu: {len(np.unique(y_encoded))}")

# === Analiza najważniejszych cech przed treningiem ===
feature_cols = [col for col in df_combined.columns if col not in ['Channel', 'Window', 'GESTURE']]

# Analiza korelacji cech z gestami
print("\n=== Analiza korelacji cech z gestami ===")
correlation_results = []

# Dla każdej cechy
for feature in feature_cols:
    # Dla każdego gestu
    for gesture in sorted(df_combined['GESTURE'].unique()):
        # Dane dla tego gestu
        gesture_data = df_combined[df_combined['GESTURE'] == gesture][feature].values
        # Dane dla innych gestów
        other_data = df_combined[df_combined['GESTURE'] != gesture][feature].values

        # Obliczanie średniej wartości cechy dla tego gestu i innych
        gesture_mean = np.mean(gesture_data)
        other_mean = np.mean(other_data)

        # Obliczanie różnicy (im większa, tym cecha lepiej rozróżnia gest)
        diff = abs(gesture_mean - other_mean)

        # Normalizacja różnicy przez odchylenie standardowe
        std = np.std(np.concatenate([gesture_data, other_data]))
        if std > 0:
            normalized_diff = diff / std
        else:
            normalized_diff = 0

        correlation_results.append((feature, gesture, normalized_diff))

# Sortowanie wyników według znaczenia
correlation_results.sort(key=lambda x: x[2], reverse=True)

# Top 10 najważniejszych kombinacji cecha-gest
print("Top 10 najważniejszych kombinacji cecha-gest:")
for feature, gesture, score in correlation_results[:10]:
    print(f"Cecha {feature} dla gestu {gesture}: {score:.4f}")

# Analiza kanałów
print("\n=== Analiza znaczenia kanałów ===")
channel_scores = {}

for channel in sorted(df_combined['Channel'].unique()):
    channel_data = df_combined[df_combined['Channel'] == channel]

    # Średnia wariancja cech dla tego kanału
    variance = 0
    for feature in feature_cols:
        # Dla każdego gestu sprawdzamy wariancję
        variances = []
        for gesture in sorted(channel_data['GESTURE'].unique()):
            gesture_channel_data = channel_data[channel_data['GESTURE'] == gesture][feature].values
            if len(gesture_channel_data) > 1:  # Potrzebujemy co najmniej 2 punktów do wariancji
                variances.append(np.var(gesture_channel_data))

        if variances:
            # Średnia wariancja dla cechy między gestami
            variance += np.mean(variances)

    # Zapisujemy średnią wariancję dla kanału
    if len(feature_cols) > 0:
        channel_scores[channel] = variance / len(feature_cols)
    else:
        channel_scores[channel] = 0

# Sortowanie kanałów według znaczenia
sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)

print("Ranking kanałów według wariancji cech:")
for channel, score in sorted_channels:
    print(f"Kanał {channel}: {score:.4f}")

# === Znaczenie okien ===
print("\n=== Analiza znaczenia okien ===")
window_scores = {}

for window in sorted(df_combined['Window'].unique()):
    window_data = df_combined[df_combined['Window'] == window]

    # Średnia wariancja cech dla tego okna
    variance = 0
    for feature in feature_cols:
        # Dla każdego gestu sprawdzamy wariancję
        variances = []
        for gesture in sorted(window_data['GESTURE'].unique()):
            gesture_window_data = window_data[window_data['GESTURE'] == gesture][feature].values
            if len(gesture_window_data) > 1:  # Potrzebujemy co najmniej 2 punktów do wariancji
                variances.append(np.var(gesture_window_data))

        if variances:
            # Średnia wariancja dla cechy między gestami
            variance += np.mean(variances)

    # Zapisujemy średnią wariancję dla okna
    if len(feature_cols) > 0:
        window_scores[window] = variance / len(feature_cols)
    else:
        window_scores[window] = 0

# Sortowanie okien według znaczenia
sorted_windows = sorted(window_scores.items(), key=lambda x: x[1], reverse=True)

print("Ranking okien według wariancji cech:")
for window, score in sorted_windows:
    print(f"Okno {window}: {score:.4f}")

# Podział na zbiór treningowy i testowy stratyfikowany według gestów
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X_seq, y_encoded, meta_seq,
    test_size=0.2, random_state=42,
    stratify=y_encoded  # Zapewnia równomierny rozkład klas w zbiorach treningowym i testowym
)

print(f"Kształt zbioru treningowego: {X_train.shape}")
print(f"Kształt zbioru testowego: {X_test.shape}")

# === Normalizacja cech ===
# Spłaszczenie danych do normalizacji
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

# Normalizacja
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Przywrócenie kształtu sekwencji
X_train = X_train_flat.reshape(X_train.shape)
X_test = X_test_flat.reshape(X_test.shape)


# === Definicja modeli dla sekwencji danych ===

# Model 1: CNN + LSTM dla sekwencji okien
def create_cnn_lstm_model(input_shape, num_classes):
    # Używamy funkcjonalnego API Keras zamiast Sequential
    inputs = Input(shape=input_shape)

    # Konwolucja na poziomie pojedynczego okna
    x = layers.TimeDistributed(layers.Conv1D(32, 3, activation='relu', padding='same'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

    x = layers.TimeDistributed(layers.Conv1D(64, 3, activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

    # Spłaszczenie cech konwolucyjnych
    x = layers.TimeDistributed(layers.Flatten())(x)

    # LSTM do analizy sekwencji okien
    x = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)

    # Warstwy klasyfikacji
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# === Wybór i kompilacja modelu ===
print("\n=== Tworzenie i kompilacja modelu ===")
# Dostosuj dane dla modelu CNN+LSTM
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2], 1)
num_classes = len(np.unique(y_encoded))

# Tworzenie modelu
model = create_cnn_lstm_model(input_shape, num_classes)

# Podsumowanie modelu
model.summary()

# Kompilacja modelu
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks do treningu ===
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

# === Trenowanie modelu ===
print("\n=== Trenowanie modelu na sekwencjach okien ===")
history = model.fit(
    X_train_reshaped, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Ewaluacja modelu ===
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f"\nDokładność modelu na zbiorze testowym: {test_acc:.4f}")

# === Przewidywanie i analiza błędów ===
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

# === Wizualizacja wyników treningu ===
plt.figure(figsize=(12, 5))

# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)

# Wykres funkcji straty
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Funkcja straty')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'training_history.png'))
plt.close()

# Macierz konfuzji
cm = confusion_matrix(y_test, y_pred_classes)
print("\nMacierz konfuzji:")

plt.figure(figsize=(15, 12))
display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=[f"Gest {i + 1}" for i in range(num_classes)])
display.plot(cmap='Blues', values_format='d')
plt.title('Macierz konfuzji')
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
plt.close()

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
target_names = [f"Gest {i + 1}" for i in range(num_classes)]
print(classification_report(y_test, y_pred_classes, target_names=target_names))

# === Analiza klas z najniższą dokładnością ===
class_accuracies = []
for i in range(num_classes):
    # Znajdź wszystkie próbki dla klasy i
    idx = np.where(y_test == i)[0]
    if len(idx) > 0:
        # Sprawdź dokładność dla tej klasy
        class_acc = np.sum(y_pred_classes[idx] == y_test[idx]) / len(idx)
        class_accuracies.append((i + 1, class_acc, len(idx)))

# Sortuj według dokładności (rosnąco)
class_accuracies.sort(key=lambda x: x[1])

print("\nDokładności dla poszczególnych gestów (od najgorszej do najlepszej):")
for gest, acc, count in class_accuracies:
    print(f"Gest {gest}: {acc:.4f} (liczba próbek: {count})")

# Wykres dokładności dla poszczególnych gestów
plt.figure(figsize=(14, 8))
gests = [x[0] for x in class_accuracies]
accs = [x[1] for x in class_accuracies]
counts = [x[2] for x in class_accuracies]

# Tworzymy kolorową mapę bazującą na dokładności
colors = plt.cm.RdYlGn(accs)

bars = plt.bar(gests, accs, color=colors)
plt.xlabel('Numer gestu')
plt.ylabel('Dokładność')
plt.title('Dokładność klasyfikacji dla poszczególnych gestów')
plt.xticks(gests)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dodajemy liczbę próbek nad każdym słupkiem
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'n={count}', ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'gesture_accuracies.png'))
plt.close()

# === Analiza znaczenia cech ===
# Tworzymy model do ekstrakcji cech
feature_extractor = Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output  # Przedostatnia warstwa przed klasyfikacją
)

# Ekstrahujemy cechy
features = feature_extractor.predict(X_test_reshaped)


# Używamy gradientów do oszacowania znaczenia cech
def compute_feature_importance():
    # Tworzymy słownik do przechowywania znaczenia cech
    feature_importance = {}

    # Dla każdej cechy w danych oryginalnych
    for i, feature_name in enumerate(feature_cols):
        importance_sum = 0

        # Dla każdego gestu
        for gesture in sorted(df_combined['GESTURE'].unique()):
            # Dane tylko dla tego gestu
            gesture_data = df_combined[df_combined['GESTURE'] == gesture][feature_name].values

            # Dane dla innych gestów
            other_data = df_combined[df_combined['GESTURE'] != gesture][feature_name].values

            if len(gesture_data) > 0 and len(other_data) > 0:
                # Obliczamy różnicę średnich znormalizowaną przez odchylenie standardowe
                mean_diff = abs(np.mean(gesture_data) - np.mean(other_data))
                std_pooled = np.sqrt((np.var(gesture_data) + np.var(other_data)) / 2)

                if std_pooled > 0:
                    effect_size = mean_diff / std_pooled
                    importance_sum += effect_size

        # Zapisujemy całkowitą ważność cechy (suma dla wszystkich gestów)
        feature_importance[feature_name] = importance_sum

    return feature_importance


# Obliczamy znaczenie cech
feature_importance = compute_feature_importance()

# Sortujemy cechy według ważności
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("\n=== Ranking znaczenia cech ===")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Wizualizacja znaczenia cech
plt.figure(figsize=(14, 10))
features = [x[0] for x in sorted_features]
importances = [x[1] for x in sorted_features]

# Normalizacja znaczenia cech do przedziału [0,1]
importances_norm = importances / max(importances)

# Wykres słupkowy z kolorem zależnym od ważności
colors = plt.cm.viridis(importances_norm)
bars = plt.barh(features, importances, color=colors)
plt.xlabel('Ważność cechy')
plt.title('Ranking ważności cech')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
plt.close()

# === Analiza znaczenia kanałów na podstawie metadanych ===
channel_accuracies = {}
channel_counts = {}

# Dla każdego przewidywania, sprawdzamy czy było poprawne i zapisujemy dla odpowiedniego kanału
for i, (pred, true, meta) in enumerate(zip(y_pred_classes, y_test, meta_test)):
    channel = meta[1]  # Kanał z metadanych

    if channel not in channel_accuracies:
        channel_accuracies[channel] = 0
        channel_counts[channel] = 0

    # Sprawdź czy przewidywanie było poprawne
    if pred == true:
        channel_accuracies[channel] += 1

    channel_counts[channel] += 1

# Obliczamy średnią dokładność dla każdego kanału
for channel in channel_accuracies:
    if channel_counts[channel] > 0:
        channel_accuracies[channel] /= channel_counts[channel]

# Sortujemy kanały według dokładności
sorted_channel_acc = sorted(channel_accuracies.items(), key=lambda x: x[1], reverse=True)

print("\n=== Ranking kanałów według dokładności klasyfikacji ===")
for channel, acc in sorted_channel_acc:
    print(f"Kanał {channel}: {acc:.4f} (liczba próbek: {channel_counts[channel]})")

# Wizualizacja znaczenia kanałów według dokładności
plt.figure(figsize=(14, 8))
channels = [x[0] for x in sorted_channel_acc]
accs = [x[1] for x in sorted_channel_acc]
counts = [channel_counts[ch] for ch in channels]

# Tworzymy kolorową mapę bazującą na dokładności
colors = plt.cm.viridis(accs)

bars = plt.bar(channels, accs, color=colors)
plt.xlabel('Numer kanału (elektrody)')
plt.ylabel('Dokładność')
plt.title('Dokładność klasyfikacji dla poszczególnych kanałów')
plt.xticks(rotation=90)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dodajemy liczbę próbek nad każdym słupkiem
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'n={count}', ha='center', va='bottom', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'channel_importance.png'))
plt.close()

# NOWA FUNKCJONALNOŚĆ: Wykres kanałów sortowanych według liczby próbek (n)
# Sortujemy kanały według liczby próbek
sorted_channels_by_count = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)

print("\n=== Ranking kanałów według liczby próbek (n) ===")
for channel, count in sorted_channels_by_count:
    acc = channel_accuracies.get(channel, 0)
    print(f"Kanał {channel}: {count} próbek, dokładność: {acc:.4f}")

plt.figure(figsize=(14, 8))
channels_by_count = [x[0] for x in sorted_channels_by_count]
counts_sorted = [x[1] for x in sorted_channels_by_count]
accs_by_count = [channel_accuracies.get(ch, 0) for ch in channels_by_count]

# Wykres słupkowy z liczbą próbek
fig, ax1 = plt.subplots(figsize=(14, 8))

# Oś pierwsza - liczba próbek (słupki)
bars = ax1.bar(channels_by_count, counts_sorted, color='lightblue', alpha=0.7)
ax1.set_xlabel('Numer kanału (elektrody)')
ax1.set_ylabel('Liczba próbek (n)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(channels_by_count)
ax1.set_xticklabels(channels_by_count, rotation=90)

# Oś druga - dokładność (linia)
ax2 = ax1.twinx()
line = ax2.plot(channels_by_count, accs_by_count, 'ro-', linewidth=2, markersize=8)
ax2.set_ylabel('Dokładność', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 1.05)

plt.title('Ranking kanałów według liczby próbek (n) z dokładnością')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'channel_importance_by_count.png'))
plt.close()

# === Analiza znaczenia okien na podstawie metadanych ===
window_accuracies = {}
window_counts = {}

# Dla każdego przewidywania, sprawdzamy czy było poprawne
for i, (pred, true, meta) in enumerate(zip(y_pred_classes, y_test, meta_test)):
    windows = meta[2]  # Lista okien z metadanych

    for window in windows:
        if window not in window_accuracies:
            window_accuracies[window] = 0
            window_counts[window] = 0

        # Sprawdź czy przewidywanie było poprawne
        if pred == true:
            window_accuracies[window] += 1

        window_counts[window] += 1

# Obliczamy średnią dokładność dla każdego okna
for window in window_accuracies:
    if window_counts[window] > 0:
        window_accuracies[window] /= window_counts[window]

# Sortujemy okna według dokładności
sorted_window_acc = sorted(window_accuracies.items(), key=lambda x: x[1], reverse=True)

print("\n=== Ranking okien według dokładności klasyfikacji ===")
for window, acc in sorted_window_acc:
    print(f"Okno {window}: {acc:.4f} (liczba próbek: {window_counts[window]})")

# Wizualizacja znaczenia okien
plt.figure(figsize=(14, 8))
windows = [x[0] for x in sorted_window_acc]
accs = [x[1] for x in sorted_window_acc]
counts = [window_counts[w] for w in windows]

# Tworzymy kolorową mapę bazującą na dokładności
colors = plt.cm.viridis(accs)

bars = plt.bar(windows, accs, color=colors)
plt.xlabel('Numer okna')
plt.ylabel('Dokładność')
plt.title('Dokładność klasyfikacji dla poszczególnych okien')
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dodajemy liczbę próbek nad każdym słupkiem
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'n={count}', ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'window_importance.png'))
plt.close()

# === Korelacja między cechami ===
# Wybieramy podzbiór danych dla analizy korelacji
sample_data = df_combined.sample(min(10000, len(df_combined)))
correlation_matrix = sample_data[feature_cols].corr().abs()  # Używamy wartości bezwzględnych korelacji

# Wizualizacja macierzy korelacji
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Macierz korelacji między cechami')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'feature_correlation.png'))
plt.close()

# NOWA FUNKCJONALNOŚĆ: Ranking cech od najmniej skorelowanych
# Obliczenie sumy korelacji dla każdej cechy (z wyłączeniem korelacji cechy z samą sobą)
feature_correlation_sums = {}
for feature in feature_cols:
    # Suma wszystkich korelacji z tą cechą, z wyjątkiem korelacji z samą sobą
    sum_corr = correlation_matrix[feature].sum() - 1  # odejmujemy 1, bo korelacja cechy z samą sobą = 1
    feature_correlation_sums[feature] = sum_corr

# Sortowanie cech według sumy korelacji (od najmniejszej do największej)
sorted_features_by_correlation = sorted(feature_correlation_sums.items(), key=lambda x: x[1])

print("\n=== Ranking cech od najmniej skorelowanych (niezależnych) do najbardziej skorelowanych ===")
for feature, corr_sum in sorted_features_by_correlation:
    print(f"{feature}: {corr_sum:.4f}")

# Wizualizacja rankingu cech według niezależności
plt.figure(figsize=(14, 10))
features_by_corr = [x[0] for x in sorted_features_by_correlation]
corr_sums = [x[1] for x in sorted_features_by_correlation]

# Normalizacja sum korelacji do przedziału [0,1]
corr_sums_norm = np.array(corr_sums) / max(corr_sums)

# Wykres słupkowy z kolorem zależnym od niezależności cechy (odwrotność korelacji)
colors = plt.cm.viridis(1 - corr_sums_norm)  # Odwracamy, żeby jaśniejsze były najmniej skorelowane
bars = plt.barh(features_by_corr, corr_sums, color=colors)
plt.xlabel('Suma korelacji z innymi cechami')
plt.title('Ranking cech według niezależności (od najmniej do najbardziej skorelowanych)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'feature_independence_ranking.png'))
plt.close()

# === Wizualizacja przestrzeni cech za pomocą t-SNE ===
# Pobieramy cechy z przedostatniej warstwy dla wizualizacji
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)

# Ekstrahujemy cechy
test_features = feature_extractor.predict(X_test_reshaped)

# Używamy t-SNE do redukcji wymiarowości
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
test_features_tsne = tsne.fit_transform(test_features)

# Wizualizacja
plt.figure(figsize=(14, 12))

# Definiujemy kolory dla każdej klasy
colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

# Dla każdej klasy rysujemy punkty
for i in range(num_classes):
    idx = np.where(y_test == i)[0]
    plt.scatter(test_features_tsne[idx, 0], test_features_tsne[idx, 1],
                color=colors[i], label=f'Gest {i + 1}', alpha=0.7, edgecolors='w', linewidth=0.5)

plt.title('Wizualizacja t-SNE przestrzeni cech modelu')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(plots_dir, 'tsne_features.png'), bbox_inches='tight')
plt.close()

# === Zapisywanie modelu ===
model.save('emg_model_cnn_lstm.h5')
print(f"\nModel został zapisany jako emg_model_cnn_lstm.h5")

print("\n=== Podsumowanie analiz ===")
print("1. Dokładność modelu:", test_acc)
print(f"2. Wszystkie wizualizacje zostały zapisane w folderze '{plots_dir}'")
print("3. Zidentyfikowano najważniejsze cechy, kanały i okna")
print("4. Utworzono dodatkowe rankingi kanałów według liczby próbek (n) i cech według niezależności")
print("5. Model został zapisany do dalszego użytku")

# Podsumowanie najważniejszych wyników w jednej tabeli
print("\n=== Tabela wyników - TOP 5 ===")
print("Top 5 najważniejszych cech:")
for feature, importance in sorted_features[:5]:
    print(f"- {feature}: {importance:.4f}")

print("\nTop 5 najbardziej niezależnych cech:")
for feature, corr in sorted_features_by_correlation[:5]:
    print(f"- {feature}: {corr:.4f}")

print("\nTop 5 najważniejszych kanałów:")
for channel, acc in sorted_channel_acc[:5]:
    print(f"- Kanał {channel}: {acc:.4f} (n={channel_counts[channel]})")

print("\nTop 5 kanałów z największą liczbą próbek:")
for channel, count in sorted_channels_by_count[:5]:
    acc = channel_accuracies.get(channel, 0)
    print(f"- Kanał {channel}: {count} próbek, dokładność: {acc:.4f}")

print("\nTop 5 najbardziej dokładnych gestów:")
sorted_by_acc = sorted(class_accuracies, key=lambda x: x[1], reverse=True)
for gest, acc, count in sorted_by_acc[:5]:
    print(f"- Gest {gest}: {acc:.4f} (liczba próbek: {count})")