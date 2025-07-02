"""
CNN DO ROZPOZNAWANIA GATUNKÓW THERAPHOSIDAE
        Paweł Grygielski(121678)
          MODEL "FROM SCRATCH"
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Rozwiązanie problemu z bibliotekami

# --- Ładowanie pakietów ---
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- AUGMENTACJE DANYCH ---
# Tworzenie generatora danych treningowych z różnymi augmentacjami,
# aby zwiększyć różnorodność obrazów i zmniejszyć overfitting.
train_datagen = ImageDataGenerator(
    rescale=1./255,           # skalowanie pikseli do zakresu [0,1]
    shear_range=0.2,          # losowe ścinanie obrazu (shear)
    zoom_range=0.2,           # losowe przybliżanie
    horizontal_flip=True,     # losowe odbicie poziome
    rotation_range=30,        # losowe rotacje do ±30 stopni
    width_shift_range=0.2,    # przesunięcie w osi X (20% szerokości)
    height_shift_range=0.2,   # przesunięcie w osi Y (20% wysokości)
    brightness_range=[0.8, 1.2], # losowa zmiana jasności
    channel_shift_range=10.0     # losowa zmiana wartości kanałów RGB
)

# Generator dla zbioru testowego
test_datagen = ImageDataGenerator(rescale=1./255)

# --- Ścieżki do danych treningowych i testowych ---
train_path = 'C:/Users/pgryg/Desktop/CNN_project/Species'
test_path = 'C:/Users/pgryg/Desktop/CNN_project/test'

# --- Generatory danych ---
# flow_from_directory pozwala wczytać obrazy posegregowane w foldery nazwane zgodnie z klasami.
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),    # skalowanie obrazów do 224x224 (wymiar wejściowy modelu)
    batch_size=32,             # liczba batchy
    class_mode='categorical'   # klasyfikacja wieloklasowa, etykiety będą w formie one-hot
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # shuffle False ważne do późniejszej analizy błędów
)

# --- Definicja modelu CNN od podstaw ---
model = models.Sequential()

# 1. Pierwsza warstwa konwolucyjna (Conv2D)
# Wyłapuje podstawowe cechy: krawędzie, tekstury itd.
model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(224,224,3))) # 32 filtry, kernel size 3x3 - 9 pikseli
model.add(layers.BatchNormalization())  # normalizacja warstw - stabilizuje i przyspiesza trening
model.add(layers.MaxPooling2D((2,2)))   # rozmiar obrazu zmniejszony o połowę, zwiększa odporność na przesunięcia

# 2. Druga warstwa konwolucyjna - do bardziej złożonych cech
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# 3. Trzecia warstwa konwolucyjna - do jeszcze bardziej złożonych cech
model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# 4. Czwarta warstwa konwolucyjna - najwieksza ilosc filtrow - najbardziej złożone cechy
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# GlobalAveragePooling2D zamiast Flatten:
# średnia globalna po każdej mapie cech, zmniejsza liczbę parametrów i zmniejsza ryzyko overfittingu
model.add(layers.GlobalAveragePooling2D())

# Warstwa w pełni połączona (gęsta) z 512 neuronami i ReLU
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())  # normalizacja po warstwie gęstej
model.add(layers.Dropout(0.5))          # dropout 50%, zapobiega nadmiernemu dopasowaniu (overfitting)

# Ostatnia warstwa klasyfikująca: 101 klas, softmax do obliczenia prawdopodobieństw klas
num_classes = 101
model.add(layers.Dense(num_classes, activation='softmax'))

# --- Kompilacja modelu ---
# Optymalizator Adam (adaptacyjny współczynnik uczenia)
# Loss: categorical_crossentropy ze względu na wieloklasową klasyfikację z one-hot
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacki do monitorowania i kontrolowania treningu ---
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',  # monitorowanie straty walidacyjnej
    patience=20,         # jeśli przez 20 epok nie będzie poprawy, trening sie zatrzyma
    restore_best_weights=True  # po zatrzymaniu przywracane są najlepsze wagi z treningu
)

model_checkpoint = callbacks.ModelCheckpoint(
    'C:/Users/pgryg/Desktop/CNN_project/best_model.h5',
    monitor='val_loss',
    save_best_only=True,  # zapis najlepszego modelu (z najmniejszym validation loss)
    verbose=1             # przy zapisie najlepszego modelu wyświetla komunikat
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,    # zmniejsza Learning Rate o połowę
    patience=5,    # po 5 epokach bez poprawy validation loss
    min_lr=1e-6,   # najnizszy mozliwy Learning Rate 
    verbose=1      # przy zmianie Learning Rate wyświetla komunikat
)

# --- Trenowanie modelu ---
epochs = 150            # 150 epok - maksymalnie tyle razy sieć przejdzie przez dane treningowe
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[early_stop, model_checkpoint, reduce_lr]
)

# --- Wizualizacja wyników treningu ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('C:/Users/pgryg/Desktop/CNN_project/training_plot.png')
plt.show()

# --- Ewaluacja modelu na zbiorze testowym ---
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')

# --- Predykcje na zbiorze testowym ---
predictions = model.predict(test_generator)
labels = list(test_generator.class_indices.keys())

# Wybór klasy o największym prawdopodobieństwie
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Macierz pomyłek: ile razy model pomylił którą klasę z którą
cm = confusion_matrix(true_classes, pred_classes)

# Liczenie błędów na klasę: suma wiersza (wszystkie predykcje tej klasy)
# minus te poprawne
errors_per_class = cm.sum(axis=1) - np.diag(cm)

# Tworzenie DataFrame z gatunkami i liczbą błędów, sortowanie malejąco
df_errors = pd.DataFrame({
    'species': labels,
    'errors': errors_per_class
}).sort_values(by='errors', ascending=False)

print("\nSpecies with the most prediction errors:")
print(df_errors.head(10))

# Zapis wyników błędów do pliku CSV
df_errors.to_csv('C:/Users/pgryg/Desktop/CNN_project/errors_per_class.csv', index=False)

# --- Zapis finalnego modelu ---
model.save('C:/Users/pgryg/Desktop/CNN_project/model_final.h5')

# --- Tworzenie pliku CSV tylko z błędnymi predykcjami ---
filenames = test_generator.filenames
filepaths = [os.path.join(test_path, fname) for fname in filenames]

df_all = pd.DataFrame({
    'file_path': filepaths,
    'true_class': true_classes,
    'pred_class': pred_classes,
    'true_label': [labels[i] for i in true_classes],
    'pred_label': [labels[i] for i in pred_classes],
})

df_mistakes = df_all[df_all['true_class'] != df_all['pred_class']]
df_mistakes.to_csv('C:/Users/pgryg/Desktop/CNN_project/test_errors_only.csv', index=False)

print("\n❌ Błędy predykcji zapisane do: test_errors_only.csv")
