import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage import color 

data_dir = r"D:\ml task\archive\signatures"

# Lists to store features and labels
X = []  # Feature vectors
y = []  # Labels (0 for real, 1 for forged)

# Parameters for HOG
hog_params = {
    "orientations": 9,  # Number of gradient bins
    "pixels_per_cell": (8, 8),  # Size of each cell
    "cells_per_block": (2, 2),  # Number of cells per block
    "block_norm": "L2-Hys"
}

# Loop through images
for label in ["full_org", "full_forg"]:
    class_dir = os.path.join(data_dir, label)
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            #print(f"Warning: Couldn't load {img_path}. Skipping this file.")
            continue
        
        # Resize to a fixed size (important for consistency)
        img = cv2.resize(img, (128, 128))

        # Extract HOG features
        hog_features = hog(img, **hog_params)

        # Store features and label
        X.append(hog_features)
        y.append(0 if label == "full_org" else 1)  # Assign labels

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Feature extraction complete! Shape of X:", X.shape)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Split Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Normalize Features (HOG features can have large values)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train SVM Classifier
svm_model = SVC(kernel="linear", C=1)  # Linear kernel is good for binary classification
svm_model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = svm_model.predict(X_test)

# Step 5: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

train_accuracy = svm_model.score(X_train, y_train) * 100
test_accuracy = svm_model.score(X_test, y_test) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")


