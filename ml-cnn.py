import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

dataset_path = r"D:\ml task\archive\signatures"  
# Paths to original and forged signature folders
original_path = os.path.join(dataset_path, "full_org")
forged_path = os.path.join(dataset_path, "full_forg")

# Fixed image size
IMG_SIZE = 128

# Function to load images and convert them into arrays
def load_images_from_folder(folder, label):
    data = []
    for filename in os.listdir(folder):  # Loop through images
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #imread gives 3d array for colorful images and 2d arraya for grayscale images
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            #objects resized as object needs to be in same size for ml models
            img = img / 255.0  # Normalize pixel values (0-1)
            # ML models learn better when inputs are in the range 0 to 1 instead of 0 to 255.
            data.append([img, label])
    return data

# Load original and forged images
original_data = load_images_from_folder(original_path, label=1)  # 1 = Original
forged_data = load_images_from_folder(forged_path, label=0)  # 0 = Forged

# Combine and shuffle dataset
dataset = original_data + forged_data
np.random.shuffle(dataset)

# Separate features (X) and labels (y)
X = np.array([i[0] for i in dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN input
y = np.array([i[1] for i in dataset])

print(f"Dataset prepared! Total images: {len(X)}")

# Splitting the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Define the CNN model
model = models.Sequential([

    # 1st Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)), 
    layers.MaxPooling2D((2, 2)),

    # 2nd Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3rd Convolutional Layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flattening Layer
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation='relu'),

    # Output Layer (Binary Classification)
    layers.Dense(1, activation='sigmoid')  
])

y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#evaluating the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Pick a random test image
index = np.random.randint(len(X_test))
test_image = X_test[index]
# Add batch dimension (since model expects batches)
test_image = np.expand_dims(test_image, axis=0)
# Get prediction
prediction = model.predict(test_image)
# Convert probability to class label
predicted_label = 1 if prediction[0][0] > 0.5 else 0
actual_label = y_test[index]
print(f"Predicted: {predicted_label}, Actual: {actual_label}")



