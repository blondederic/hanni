import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# data engineering
def create_directories():
    os.makedirs('data/hanni', exist_ok=True)
    os.makedirs('data/others', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

# data preprocessing
def preprocess_images(image_folder, img_size):
    images = []
    labels = []
    for label in ['hanni', 'others']:
        path = os.path.join(image_folder, label)
        if not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            continue
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            if img_array is None:
                print(f"Failed to read image: {img_path}")
                continue
            try:
                img_array = cv2.resize(img_array, (img_size, img_size))
                images.append(img_array / 255.0) 
                labels.append(1 if label == 'hanni' else 0)  
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    return np.array(images), np.array(labels)

# setup
create_directories()
img_size = 128
image_folder = 'data'
images, labels = preprocess_images(image_folder, img_size)

if len(images) == 0:
    print("No images were loaded. Please check your image directories and ensure they contain valid images.")
else:
    print(f"Loaded {len(images)} images.")

# training and testing split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# data augmentation to improve generalization
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# cnn model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_size, img_size, 3)),
    data_augmentation,  

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# modeling compiling and fitting
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
              loss='binary_crossentropy', metrics=['recall', 'accuracy', 'precision'])


# Early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

# train
model.fit(X_train, y_train, epochs=30, batch_size = 64, validation_data=(X_test, y_test), callbacks=[callback])

# test model eval
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# save model
model.save("hanni_classifier_model.keras")
print("Model saved as 'hanni_classifier_model.keras'")

# data augmentation to generate more images - image data gen
# add the face detection preproccesing 
# trasnfer learning - other base models, mobilenetv2