import nltk
import random
import numpy as np
import pickle
import json
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from keras.callbacks import EarlyStopping

# Downloads
nltk.download('punkt')
nltk.download('wordnet')

# Early stopping
earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=150, verbose=1, restore_best_weights=True)

lemmatizer = WordNetLemmatizer()

# Data processing
words = []
classes = []
documents = []
ignore_words = ['?', '!']

with open("/content/merged_dataset_intents.json") as file:
    intents = json.load(file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training preparation
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1 if w in pattern_words else 0)  # Ensure consistent length with 'words'

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

# Convert to numpy arrays
train_x = np.array([np.array(x) for x, _ in training])  # Ensure all are the same length
train_y = np.array([np.array(y) for _, y in training])

# Model creation
model2 = Sequential()
model2.add(Dense(256, activation='relu', input_shape=(len(train_x[0]),)))
model2.add(Dropout(0.3))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training
hist3 = model2.fit(np.array(train_x), np.array(train_y), epochs=1500, batch_size=64, verbose=1, callbacks=[earlystop])

import matplotlib.pyplot as plt

# Plot accuracy vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(hist3.history['accuracy'], label='Training Accuracy', color='blue')
plt.title('Model Accuracy vs Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# Save model
model2.save('chatbot_model.h5')
print("Model created with accuracy: {:.2f}".format(max(hist3.history['accuracy'])))
