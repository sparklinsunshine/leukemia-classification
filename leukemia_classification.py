import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIG ---
IMAGE_SIZE = (64, 64)
DATASET_PATH = "D:/kinnudata/clg_books/sem 6/ml/leukemia/C-NMC_Leukemia/training_data"
LABELS = {'hem': 0, 'all': 1}

# --- Load dataset ---
def load_dataset(dataset_path):
    X, y = [], []
    for fold in ['fold_0', 'fold_1', 'fold_2']:
        fold_path = os.path.join(dataset_path, fold)
        if not os.path.exists(fold_path):
            continue
        for label_folder in os.listdir(fold_path):
            if label_folder not in LABELS:
                continue
            folder_path = os.path.join(fold_path, label_folder)
            for file in tqdm(os.listdir(folder_path), desc=f"{fold}/{label_folder}"):
                if not file.lower().endswith(".bmp"):
                    continue
                img = load_img(os.path.join(folder_path, file),
                               target_size=IMAGE_SIZE,
                               color_mode='grayscale')
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(LABELS[label_folder])
    return np.array(X), np.array(y)

X, y = load_dataset(DATASET_PATH)
print(f"Loaded {len(X)} images. Shape: {X.shape}")

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# --- Data augmentation ---
datagen_train = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen_train.fit(X_train)

datagen_val = ImageDataGenerator()
datagen_val.fit(X_val)

# --- CNN Model (Functional API) ---
inputs = Input(shape=(64, 64, 1))
x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(inputs)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)  # Reduced dropout

x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)  # Reduced dropout

x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)  # Added layer
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)  # Slightly higher dropout for deeper layer

x = Flatten()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='feature_layer')(x)  # Increased units
x = Dropout(0.4)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Early stopping ---
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)  # Increased patience

# --- Train CNN ---
history = model.fit(
    datagen_train.flow(X_train, y_train, batch_size=32),
    validation_data=datagen_val.flow(X_val, y_val, batch_size=32),
    epochs=50,  # Increased epochs
    callbacks=[early_stop],
    verbose=1
)

# --- Feature extraction for PCA + SVM ---
feature_model = Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)

features_train = feature_model.predict(X_train)
features_test = feature_model.predict(X_test)

# --- PCA ---
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(features_train)
X_test_pca = pca.transform(features_test)

# --- SVM Classifier ---
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)

# --- Evaluation ---
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['hem', 'all']))

print(f"Train Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Val Accuracy:   {history.history['val_accuracy'][-1] * 100:.2f}%")
print(f"Test Accuracy (SVM on PCA):  {accuracy_score(y_test, y_pred) * 100:.2f}%")

# --- Accuracy Plot ---
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Training vs Validation Accuracy')
plt.legend()
plt.grid()
plt.show()
