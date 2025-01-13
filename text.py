# Import necessary libraries
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.utils import class_weight

# Function to preprocess the data
def preprocess(file_path):
    data = pd.read_csv(file_path, sep=';')
    # Map emotions to binary labels: 1 for 'joy', 'love', 'surprise', else 0
    happy_emotions = ['joy', 'love', 'surprise']
    data['hos'] = data['emotion'].apply(lambda x: 1 if x in happy_emotions else 0)
    return data

# Define the dataset directory
dataset_dir = '/content/'  # Adjust if you're using Google Drive

# Construct paths to the training and validation dataset files
train_path = os.path.join(dataset_dir, 'train.txt')
val_path = os.path.join(dataset_dir, 'val.txt')

# Load and preprocess the datasets
train_data = preprocess(train_path)
val_data = preprocess(val_path)

# Custom Layer to wrap the TensorFlow Hub Layer
class CustomHubLayer(tf.keras.layers.Layer):
    def __init__(self, model_url, **kwargs):
        super(CustomHubLayer, self).__init__(**kwargs)
        self.hub_layer = hub.KerasLayer(model_url, output_shape=[512], input_shape=[], dtype=tf.string, trainable=True)

    def call(self, inputs):
        return self.hub_layer(inputs)

# Specify the URL to the pre-trained embedding model (BERT-based or Universal Sentence Encoder)
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

# Build the model using the functional API
inputs = tf.keras.Input(shape=[], dtype=tf.string)
x = CustomHubLayer(model_url)(inputs)

# Add layers with regularization and batch normalization
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Display model architecture
model.summary()

# Compile the model with AdamW optimizer and a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),  # Tuned learning rate
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.metrics.BinaryAccuracy(name='accuracy')]
)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data['hos']),
    y=train_data['hos']
)
class_weights = dict(enumerate(class_weights))

# Early stopping callback to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# Train the model while capturing the training history
history = model.fit(
    x=train_data['text'],
    y=train_data['hos'],
    epochs=50,  # Increased epochs for better training
    batch_size=256,  # Adjust batch size based on GPU/CPU resources
    validation_data=(val_data['text'], val_data['hos']),
    verbose=1,
    callbacks=[early_stopping],  # Early stopping callback
    class_weight=class_weights  # Add class weights
)

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_data['text'], val_data['hos'], verbose=1)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Binary Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Binary Crossentropy Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

def get_user_input_predictions(model):
    print("\nPlease describe your overall feeling:")
    user_input = input("Enter your feeling: ")

    # Convert user input into a list of strings for prediction
    user_input_list = [user_input]  # List of one string

    # Convert the list to a TensorFlow tensor for prediction
    user_input_tensor = tf.convert_to_tensor(user_input_list)

    # Predict using the model
    logits = model.predict(user_input_tensor)

    # Apply sigmoid to convert logits to probabilities
    probabilities = tf.nn.sigmoid(logits).numpy()

    # Calculate the mean probability as the mental health score
    score = np.mean(probabilities) * 100  # Scale to percentage

    # Provide feedback based on the score
    print(f'\nYour mental health score is: {score:.2f}%')
    if score < 25:
        print("You are going through a challenging phase in life. Consider seeking support.")
    elif score < 50:
        print("You're facing some difficulties, but there is room for improvement.")
    elif score < 75:
        print("Your mental health is generally good, but strive for maintenance.")
    else:
        print("Your mental health looks excellent! Keep enjoying life.")

# Call the function to get user input predictions
get_user_input_predictions(model)
