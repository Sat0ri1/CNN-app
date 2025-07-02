"""
CNN DO ROZPOZNAWANIA GATUNKÓW THERAPHOSIDAE
        Paweł Grygielski(121678)
     MODEL PRETRENOWANY Z INCEPTIONV3
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Ścieżki do zbiorów treningowego i testowego ---
train_path = 'C:/Users/pgryg/Desktop/CNN_project/Species'
test_path = 'C:/Users/pgryg/Desktop/CNN_project/test'

# --- Generatory obrazów ---
# Dla InceptionV3 wymagane jest preprocessing_function=preprocess_input,
# które normalizuje obrazy tak jak podczas treningu InceptionV3 na ImageNet.

train_datagen_inception = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # preprocessowanie pod InceptionV3
    shear_range=0.2,                         # losowe przesunięcie obrazu wzdłuż osi X
    zoom_range=0.2,                          # losowe przybliżanie obrazu
    horizontal_flip=True,                    # losowe odbicia lustrzane poziome
    rotation_range=30,                       # losowe rotacje do ±30 stopni
    width_shift_range=0.2,                   # losowe przesunięcia w osi X (20% szerokości)
    height_shift_range=0.2,                  # losowe przesunięcia w osi Y (20% wysokości)
    brightness_range=[0.8, 1.2],             # losowa zmiana jasności obrazu
    channel_shift_range=10.0                  # losowe przesunięcie wartości kanałów RGB
)

test_datagen_inception = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Tworzenie generatora danych treningowych
train_generator_inception = train_datagen_inception.flow_from_directory(
    train_path,
    target_size=(299, 299),  # InceptionV3 wymaga wejścia 299x299
    batch_size=32,
    class_mode='categorical' # klasyfikacja wieloklasowa - one-hot encoding etykiet
)

# Tworzenie generatora danych testowych - bez shuffle, aby zachować kolejność
test_generator_inception = test_datagen_inception.flow_from_directory(
    test_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- Budowa modelu ---
# Wykorzystuje gotowy model InceptionV3 jako ekstraktor cech (base_model),
# wczytując wagi wytrenowane na ImageNet.
# include_top=False oznacza, że nie używa domyślnej gęstej warstwy klasyfikacyjnej InceptionV3.

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
base_model.trainable = False  # zamrażenie warstw bazowych - nie będą trenowane

# Tworzenie modelu sekwencyjnego na bazie ekstraktora cech
model_inception = models.Sequential([
    base_model,  # ekstrakcja cech z InceptionV3

    # GlobalAveragePooling2D - zamiast spłaszczać feature mapy, 
    # uśrednia wartości cech w każdej mapie, zmniejszając liczbę parametrów i przeciwdziałając przeuczeniu.
    layers.GlobalAveragePooling2D(),

    # Gęsta warstwa ukryta z 512 neuronami i aktywacją ReLU,
    # pozwalająca modelowi uczyć się bardziej złożonych reprezentacji.
    layers.Dense(512, activation='relu'),

    # Normalizacja batchy - pomaga stabilizować i przyspieszać trening
    layers.BatchNormalization(),

    # Dropout 0.5 - zapobiega przeuczeniu przez losowe "wyłączanie" połowy neuronów podczas treningu
    layers.Dropout(0.5),

    # Ostatnia warstwa gęsta z 101 neuronami (liczba klas),
    # aktywacja softmax zapewnia, że wyjścia to prawdopodobieństwa klas.
    layers.Dense(101, activation='softmax')
])

# --- Kompilacja modelu ---
# Używacie optymalizatora Adam, który adaptuje współczynnik uczenia.
# Funkcja straty categorical_crossentropy jest standardowa dla klasyfikacji wieloklasowej.
model_inception.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']  # podczas treningu monitorujemy dokładność
)

# --- Definicja callbacków ---
# EarlyStopping - zatrzymuje trening jeśli przez 20 epok nie poprawi się 'val_loss'
early_stop_inception = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# ModelCheckpoint - zapisuje najlepszy model na dysku, monitorując 'val_loss'
model_checkpoint_inception = callbacks.ModelCheckpoint(
    'C:/Users/pgryg/Desktop/CNN_project/best_model_inception.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ReduceLROnPlateau - zmniejsza LR jeśli 'val_loss' przestaje się poprawiać przez 5 epok
reduce_lr_inception = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# --- Trenowanie modelu ---
epochs = 150
history_inception = model_inception.fit(
    train_generator_inception,
    epochs=epochs,
    validation_data=test_generator_inception,
    callbacks=[early_stop_inception, model_checkpoint_inception, reduce_lr_inception]
)

# --- FINE-TUNING INCEPTIONV3 ---
# Odblokowujemy część warstw modelu bazowego w celu dalszego dopasowania do konkretnego zbioru danych.
# To tzw. fine-tuning – dalsze dostrajanie wag wcześniej wytrenowanego modelu.
# Te warstwy na wcześniejszych etapach treningu były zamrożone w 68 linijce poprzez base_model.trainable = False 

base_model.trainable = True  # Odblokowuje wszystkie warstwy bazowe

# Sprawdzenie ile warstw ma model bazowy
print(f"\n➡️ Liczba warstw w modelu bazowym InceptionV3: {len(base_model.layers)}")

# Zamraża pierwsze 250 warstw – nie będą trenowane.
# Pozwala to zachować ogólne cechy wyuczone na ImageNet, 
# jednocześnie umożliwiając naukę bardziej specyficznych cech dla nowego zbioru danych w dalszych warstwach.
for layer in base_model.layers[:250]:
    layer.trainable = False

# Ponowna kompilacja modelu z mniejszym learning rate
# (ważne przy fine-tuningu, aby nie "zepsuć" wcześniej wyuczonych wag dużymi zmianami).
model_inception.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback do zapisywania najlepszego modelu podczas fine-tuningu
model_checkpoint_finetune = callbacks.ModelCheckpoint(
    'C:/Users/pgryg/Desktop/CNN_project/best_model_inception_finetuned.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# --- Trening modelu z odblokowanymi warstwami (fine-tuning) ---
# Kontynuujemy trenowanie przez dodatkowe 150 epok, teraz ucząc model bazowy InceptionV3.
fine_tune_epochs = 150
history_finetune = model_inception.fit(
    train_generator_inception,
    epochs=fine_tune_epochs,
    validation_data=test_generator_inception,
    callbacks=[early_stop_inception, model_checkpoint_finetune, reduce_lr_inception]
)

# --- Wizualizacja wyników po fine-tuningu ---
plt.figure(figsize=(12,5))

# Dokładność
plt.subplot(1,2,1)
plt.plot(history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy after Fine-tuning (InceptionV3)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Strata
plt.subplot(1,2,2)
plt.plot(history_finetune.history['loss'], label='Train Loss')
plt.plot(history_finetune.history['val_loss'], label='Val Loss')
plt.title('Model Loss after Fine-tuning (InceptionV3)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('C:/Users/pgryg/Desktop/CNN_project/training_plot_inception_finetuned.png')
plt.show()

# --- Ewaluacja po fine-tuningu ---
test_loss_ft, test_acc_ft = model_inception.evaluate(test_generator_inception)
print(f'\n✅ Test accuracy after fine-tuning: {test_acc_ft:.4f}')

# --- Predykcje i macierz pomyłek ---
predictions_ft = model_inception.predict(test_generator_inception)
pred_classes_ft = np.argmax(predictions_ft, axis=1)
true_classes = test_generator_inception.classes

labels = list(test_generator_inception.class_indices.keys())
cm_ft = confusion_matrix(true_classes, pred_classes_ft)
errors_ft = cm_ft.sum(axis=1) - np.diag(cm_ft)

df_errors_ft = pd.DataFrame({
    'species': labels,
    'errors': errors_ft
}).sort_values(by='errors', ascending=False)

df_errors_ft.to_csv('C:/Users/pgryg/Desktop/CNN_project/errors_per_class_inception_finetuned.csv', index=False)
print("\n📊 Zapisano tabelę błędów po fine-tuningu.")

# --- Zapis błędnych predykcji po fine-tuningu ---
filenames = test_generator_inception.filenames
filepaths = [os.path.join(test_path, fname) for fname in filenames]

df_all_ft = pd.DataFrame({
    'file_path': filepaths,
    'true_class': true_classes,
    'pred_class': pred_classes_ft,
    'true_label': [labels[i] for i in true_classes],
    'pred_label': [labels[i] for i in pred_classes_ft],
})

df_mistakes_ft = df_all_ft[df_all_ft['true_class'] != df_all_ft['pred_class']]
df_mistakes_ft.to_csv('C:/Users/pgryg/Desktop/CNN_project/test_errors_only_inception_finetuned.csv', index=False)
print("❌ Błędy predykcji (fine-tuning) zapisane do: test_errors_only_inception_finetuned.csv")

# --- Zapis ostatecznego modelu ---
model_inception.save('C:/Users/pgryg/Desktop/CNN_project/model_final_inception_finetuned.h5')