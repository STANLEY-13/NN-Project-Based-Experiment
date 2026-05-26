# Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:

```py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical

# Normalize pixel values to range [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten 28x28 images into 784 features
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert labels to categorical format
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Split training data into training + validation
x_train, x_val, y_train_cat, y_val_cat = train_test_split(
    x_train, y_train_cat, test_size=0.2, random_state=42
)

model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(256, activation='relu', input_shape=(784,)))

# Hidden Layer 2
model.add(Dense(128, activation='relu'))

# Dropout Layer
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display Model Summary
model.summary()

history = model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val_cat)
)

test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes))

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix\nSTANLEY - MLP MNIST Project")

plt.show()

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.suptitle("STANLEY - MLP MNIST Project", fontsize=14)

plt.legend()
plt.show()

# Loss Graph
plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.suptitle("STANLEY - MLP MNIST Project", fontsize=14)

plt.legend()
plt.show()

misclassified = np.where(y_pred_classes != y_test)[0]

plt.figure(figsize=(12,8))

for i in range(9):
    index = misclassified[i]

    plt.subplot(3,3,i+1)
    plt.imshow(x_test[index].reshape(28,28), cmap='gray')

    plt.title(f"True: {y_test[index]} | Pred: {y_pred_classes[index]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## Output:
<img width="658" height="568" alt="image" src="https://github.com/user-attachments/assets/16bceac4-cee2-4e15-8738-1e8f7a30ced4" />
<img width="855" height="498" alt="image" src="https://github.com/user-attachments/assets/fa856845-8d60-4ffe-a82a-cfa8ad056629" />
<img width="855" height="498" alt="image" src="https://github.com/user-attachments/assets/b5c397f4-4f97-4faa-8f8b-6dcf7f1eb40e" />
<img width="972" height="790" alt="image" src="https://github.com/user-attachments/assets/0dfa2835-f08e-486b-88d6-8b61f5a9d9c2" />


## Result:

Result

The Multilayer Perceptron (MLP) model was successfully trained using the MNIST handwritten digit dataset. The model achieved high classification accuracy on the test dataset and effectively recognized handwritten digits from 0–9. The training and validation graphs showed good learning performance with minimal overfitting.
