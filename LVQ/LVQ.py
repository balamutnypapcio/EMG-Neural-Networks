import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


class AdvancedLVQ(BaseEstimator, ClassifierMixin):
    """
    An advanced Learning Vector Quantization implementation with:
    - Multiple prototypes per class
    - Adaptive learning rate
    - Momentum
    - Relevance learning (GRLVQ-like approach)
    - Learning progress tracking
    """

    def __init__(self, n_prototypes_per_class=3, initial_learning_rate=0.1,
                 final_learning_rate=0.001, max_iter=1000, momentum=0.9,
                 window_size=5, relevance_learning_rate=0.05, random_state=None):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.max_iter = max_iter
        self.momentum = momentum
        self.window_size = window_size
        self.relevance_learning_rate = relevance_learning_rate
        self.random_state = random_state

        # Components to be learned
        self.prototypes_ = None
        self.prototype_labels_ = None
        self.feature_relevances_ = None
        self.history_ = {'accuracy': [], 'learning_rate': [], 'epochs': []}

    def _initialize_prototypes(self, X, y):
        """Initialize prototypes using k-means like approach for each class"""
        np.random.seed(self.random_state)
        unique_labels = np.unique(y)
        prototypes = []
        prototype_labels = []

        for label in unique_labels:
            class_samples = X[y == label]
            n_samples = min(len(class_samples), self.n_prototypes_per_class)

            if n_samples < self.n_prototypes_per_class:
                # If we have fewer samples than prototypes, use all samples and duplicate some
                indices = np.random.choice(len(class_samples), self.n_prototypes_per_class, replace=True)
                selected_prototypes = class_samples[indices]
            else:
                # Initialize prototypes using stratified selection
                selected_indices = np.random.choice(len(class_samples), self.n_prototypes_per_class, replace=False)
                selected_prototypes = class_samples[selected_indices]

                # Refine initial prototypes with a mini k-means like approach
                for _ in range(20):  # More iterations to refine initial positions
                    # Assign each sample to the closest prototype
                    assignments = []
                    for sample in class_samples:
                        distances = np.sqrt(np.sum((selected_prototypes - sample) ** 2, axis=1))
                        closest_idx = np.argmin(distances)
                        assignments.append(closest_idx)

                    assignments = np.array(assignments)

                    # Update prototypes as the mean of assigned samples
                    for i in range(len(selected_prototypes)):
                        assigned_samples = class_samples[assignments == i]
                        if len(assigned_samples) > 0:
                            selected_prototypes[i] = np.mean(assigned_samples, axis=0)

            for prototype in selected_prototypes:
                prototypes.append(prototype)
                prototype_labels.append(label)

        return np.array(prototypes), np.array(prototype_labels)

    def _calculate_learning_rate(self, epoch):
        """Calculate adaptive learning rate based on epoch - using more aggressive decay"""
        return self.initial_learning_rate * np.exp(-5 * epoch / self.max_iter)

    def _calculate_distance(self, x, prototype):
        """Calculate weighted Euclidean distance"""
        return np.sqrt(np.sum(self.feature_relevances_ * (x - prototype) ** 2))

    def fit(self, X, y):
        """Fit the LVQ model to the data with progress tracking"""
        # Initialize prototypes and relevances
        self.prototypes_, self.prototype_labels_ = self._initialize_prototypes(X, y)
        self.feature_relevances_ = np.ones(X.shape[1]) / X.shape[1]  # Equal weights initially

        # Initialize variables for momentum
        prev_updates = np.zeros_like(self.prototypes_)
        prev_relevance_updates = np.zeros_like(self.feature_relevances_)

        # Initialize variables for monitoring
        self.history_['accuracy'] = []
        self.history_['learning_rate'] = []
        self.history_['relevances'] = []
        self.history_['epochs'] = []

        # Training loop with progress bar
        for epoch in tqdm(range(self.max_iter), desc="Training LVQ"):
            # Calculate current learning rate with decay
            current_lr = self._calculate_learning_rate(epoch)
            self.history_['learning_rate'].append(current_lr)

            # Shuffle the training data
            indices = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Process mini-batches
            batch_size = min(self.window_size, len(X))
            for i in range(0, len(X_shuffled), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                prototype_updates = np.zeros_like(self.prototypes_)
                relevance_updates = np.zeros_like(self.feature_relevances_)

                for j, (sample, label) in enumerate(zip(X_batch, y_batch)):
                    # Find the closest correct and wrong prototypes
                    correct_dist = []
                    wrong_dist = []
                    correct_idx = []
                    wrong_idx = []

                    for k, (prototype, proto_label) in enumerate(zip(self.prototypes_, self.prototype_labels_)):
                        dist = self._calculate_distance(sample, prototype)
                        if proto_label == label:
                            correct_dist.append(dist)
                            correct_idx.append(k)
                        else:
                            wrong_dist.append(dist)
                            wrong_idx.append(k)

                    # Handle edge cases
                    if not correct_dist or not wrong_dist:
                        continue

                    # Find closest correct and wrong prototypes
                    closest_correct_idx = correct_idx[np.argmin(correct_dist)]
                    closest_wrong_idx = wrong_idx[np.argmin(wrong_dist)]

                    # Update closest correct prototype (move closer)
                    diff_correct = sample - self.prototypes_[closest_correct_idx]
                    prototype_updates[closest_correct_idx] += current_lr * diff_correct

                    # Update closest wrong prototype (move away)
                    diff_wrong = sample - self.prototypes_[closest_wrong_idx]
                    prototype_updates[closest_wrong_idx] -= current_lr * diff_wrong

                    # Update feature relevances (GRLVQ approach)
                    d_correct = self._calculate_distance(sample, self.prototypes_[closest_correct_idx])
                    d_wrong = self._calculate_distance(sample, self.prototypes_[closest_wrong_idx])

                    # Relative distance difference for relevance learning
                    d_sum = d_correct + d_wrong
                    if d_sum > 0:  # Avoid division by zero
                        mu = (d_correct - d_wrong) / d_sum

                        # Relevance gradient
                        relevance_gradient = np.zeros_like(self.feature_relevances_)
                        for feature in range(len(self.feature_relevances_)):
                            # Partial derivative with respect to each feature relevance
                            correct_term = (sample[feature] - self.prototypes_[closest_correct_idx][feature]) ** 2
                            wrong_term = (sample[feature] - self.prototypes_[closest_wrong_idx][feature]) ** 2

                            relevance_gradient[feature] = (
                                    (wrong_term * d_correct - correct_term * d_wrong) /
                                    (d_sum ** 2)
                            )

                        # Apply relevance update with sigmoid to stabilize
                        relevance_updates += self.relevance_learning_rate * relevance_gradient

                # Apply momentum to updates
                prototype_updates = self.momentum * prev_updates + prototype_updates
                relevance_updates = self.momentum * prev_relevance_updates + relevance_updates

                # Update prototypes and relevances
                self.prototypes_ += prototype_updates

                # Update relevances while ensuring constraints (sum to 1, non-negative)
                self.feature_relevances_ += relevance_updates
                self.feature_relevances_ = np.maximum(self.feature_relevances_, 0.001)  # Ensure positive values
                self.feature_relevances_ /= np.sum(self.feature_relevances_)  # Normalize to sum to 1

                # Store updates for momentum
                prev_updates = prototype_updates
                prev_relevance_updates = relevance_updates

            # Calculate and store accuracy every few epochs
            if epoch % 10 == 0 or epoch == self.max_iter - 1:
                self.history_['epochs'].append(epoch)
                y_pred = self.predict(X)
                acc = accuracy_score(y, y_pred)
                self.history_['accuracy'].append(acc)

                # Store current relevances
                self.history_['relevances'].append(self.feature_relevances_.copy())

                # Early stopping if accuracy is high enough
                if acc > 0.95:
                    print(f"Early stopping at epoch {epoch} with accuracy {acc:.4f}")
                    break

        return self

    def predict(self, X):
        """Predict class labels for samples in X using weighted distances"""
        if self.prototypes_ is None:
            raise ValueError("Model has not been fitted yet.")

        predictions = []
        for sample in X:
            distances = np.array([self._calculate_distance(sample, p) for p in self.prototypes_])
            closest_idx = np.argmin(distances)
            predictions.append(self.prototype_labels_[closest_idx])

        return np.array(predictions)


# --- Wczytywanie danych ---
df = pd.read_csv('emg.csv', header=None)
X = df.iloc[:, :80].values
y = df.iloc[:, 80].values

# --- Etykiety jako liczby całkowite ---
le = LabelEncoder()
y = le.fit_transform(y)

# --- Normalizacja ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Więcej informacji o danych ---
n_classes = len(np.unique(y))
class_counts = np.bincount(y)
print(f"Liczba klas: {n_classes}")
print(f"Liczba przykładów dla każdej klasy: {class_counts}")
print(f"Całkowita liczba przykładów: {len(y)}")

# --- Podział na zbiór treningowy i testowy ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model AdvancedLVQ z lepszymi parametrami ---
model = AdvancedLVQ(
    n_prototypes_per_class=10,  # Znacznie więcej prototypów na klasę
    initial_learning_rate=0.2,  # Wyższy początkowy współczynnik uczenia
    final_learning_rate=0.001,
    max_iter=30,  # Więcej iteracji
    momentum=0.9,
    window_size=32,  # Większy batch
    relevance_learning_rate=0.1,  # Mocniejsze uczenie istotności cech
    random_state=42
)

model.fit(X_train, y_train)

# --- Ocena na zbiorze treningowym i testowym ---
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Dokładność na zbiorze treningowym: {train_acc:.4f}")

y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Dokładność na zbiorze testowym: {test_acc:.4f}")

# --- Wykresy ---
# 1. Wykres postępu nauki
plt.figure(figsize=(15, 10))

# Wykres dokładności - używając zapisanych epok
plt.subplot(2, 2, 1)
plt.plot(model.history_['epochs'], model.history_['accuracy'])
plt.title('Dokładność podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.grid(True)

# Wykres współczynnika uczenia
plt.subplot(2, 2, 2)
plt.plot(range(len(model.history_['learning_rate'])), model.history_['learning_rate'])
plt.title('Zmiana współczynnika uczenia')
plt.xlabel('Epoka')
plt.ylabel('Współczynnik uczenia')
plt.grid(True)

# Wykres ważności cech (co 5 punktów historii)
plt.subplot(2, 2, 3)
if len(model.history_['relevances']) > 0:
    epochs_to_plot = np.linspace(0, len(model.history_['relevances']) - 1, min(5, len(model.history_['relevances'])),
                                 dtype=int)
    for i in epochs_to_plot:
        epoch_num = model.history_['epochs'][i]
        plt.plot(model.history_['relevances'][i], label=f'Epoka {epoch_num}')
    plt.title('Ewolucja ważności cech')
    plt.xlabel('Indeks cechy')
    plt.ylabel('Ważność')
    plt.legend()
    plt.grid(True)

# 2. Ranking ważności czujników
feature_importance = model.feature_relevances_

# 80 cech = 8 czujników po 10 cech
electrode_importance = []
for i in range(8):
    chunk = feature_importance[i * 10:(i + 1) * 10]
    electrode_importance.append(np.sum(chunk))

electrode_importance = np.array(electrode_importance)
sorted_idx = np.argsort(electrode_importance)[::-1]

plt.subplot(2, 2, 4)
plt.bar(range(1, 9), electrode_importance[sorted_idx])
plt.xticks(range(1, 9), (sorted_idx + 1))
plt.xlabel("Numer elektrody (czujnika)")
plt.ylabel("Suma ważności cech")
plt.title("Ranking ważności czujników - AdvancedLVQ")
plt.grid(True)

plt.tight_layout()
plt.savefig('lvq_learning_progress.png')
plt.show()

# Dodatkowy wykres - mapa ciepła ważności cech
plt.figure(figsize=(12, 6))
feature_importance_matrix = np.zeros((8, 10))
for i in range(8):
    for j in range(10):
        feature_importance_matrix[i, j] = feature_importance[i * 10 + j]

plt.imshow(feature_importance_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Ważność cechy')
plt.xlabel('Indeks cechy dla czujnika')
plt.ylabel('Numer czujnika')
plt.title('Mapa ciepła ważności cech dla każdego czujnika')
plt.xticks(range(10))
plt.yticks(range(8), range(1, 9))
plt.tight_layout()
plt.savefig('feature_importance_heatmap.png')
plt.show()

# Wykres macierzy pomyłek
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Przewidywana klasa')
plt.ylabel('Prawdziwa klasa')
plt.title('Macierz pomyłek')
plt.savefig('confusion_matrix.png')
plt.show()