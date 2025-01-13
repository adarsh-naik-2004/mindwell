import os
import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
# from tensorflow import keras
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Concatenate, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

class MultiModalMentalHealthModel:
    def __init__(self, max_words=10000, max_length=100, n_mfcc=40):
        self.max_words = max_words
        self.max_length = max_length
        self.n_mfcc = n_mfcc
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.scaler = StandardScaler()
        self.model = self.build_model()

    def preprocess_text(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        return pad_sequences(sequences, maxlen=self.max_length)[0]

    def extract_mfcc(self, filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    def build_model(self):
        # Text model
        text_input = Input(shape=(self.max_length,))
        text_layer = Embedding(self.max_words, 32, input_length=self.max_length)(text_input)
        text_layer = Conv1D(64, 5, activation='relu')(text_layer)
        text_layer = MaxPooling1D(pool_size=2)(text_layer)
        text_layer = Conv1D(32, 3, activation='relu')(text_layer)
        text_layer = GlobalAveragePooling1D()(text_layer)
        text_output = Dense(16, activation='relu')(text_layer)

        # Audio model
        audio_input = Input(shape=(self.n_mfcc,))
        audio_layer = Dense(64, activation='relu')(audio_input)
        audio_layer = BatchNormalization()(audio_layer)
        audio_layer = Dense(32, activation='relu')(audio_layer)
        audio_layer = BatchNormalization()(audio_layer)
        audio_output = Dense(16, activation='relu')(audio_layer)

        # Combine text and audio models
        combined = Concatenate()([text_output, audio_output])
        combined = Dense(32, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(16, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        output = Dense(1, activation='sigmoid')(combined)

        # Create the combined model
        model = Model(inputs=[text_input, audio_input], outputs=output)
        return model

    def train(self, text_data, audio_data, labels, validation_split=0.2, batch_size=32, epochs=50):
        self.tokenizer.fit_on_texts(text_data)
        X_text = np.array([self.preprocess_text(text) for text in text_data])
        X_audio = np.array([self.extract_mfcc(file) for file in audio_data])
        self.scaler.fit(X_audio)
        X_audio = self.scaler.transform(X_audio)

        X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
            X_text, X_audio, labels, test_size=validation_split, random_state=42, stratify=labels
        )

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            [X_text_train, X_audio_train], y_train,
            validation_data=([X_text_test, X_audio_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

        return history

    def predict(self, text, audio_file):
        text_processed = self.preprocess_text(text)
        audio_features = self.extract_mfcc(audio_file)
        audio_features_scaled = self.scaler.transform([audio_features])
        prediction = self.model.predict([np.array([text_processed]), audio_features_scaled])
        return prediction[0][0]