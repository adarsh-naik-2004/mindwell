# MindWell 
MindWell is an intelligent and compassionate web application dedicated to providing accessible and comprehensive mental health support. It pioneers a multimodal approach, integrating conversational AI with advanced machine learning to offer a holistic understanding of a user's mental well-being. By analyzing not just *what* you say, but *how* you say it, MindWell aims to provide more nuanced and personalized support.

---

## Features

* **Secure User Authentication:** Safe and secure login and registration system to protect user privacy.
* **Interactive Chatbot:** An NLTK and TensorFlow-powered chatbot, trained on a diverse dataset of conversational intents, provides a safe space for users to express themselves and receive immediate, empathetic support.
* **Multimodal Mental Health Assessment:** A groundbreaking assessment tool that goes beyond traditional questionnaires. It analyzes:
    * **Textual Input:** User's written responses to the questionnaire.
    * **Vocal Tone Analysis (Planned):** Analysis of pitch, tone, and speech patterns from recorded audio responses to gauge emotional state.
    * **Facial Expression Recognition (Planned):** Utilizes the user's webcam to analyze facial expressions and micro-expressions for a deeper understanding of their emotional state.
* **Personalized Dashboard:** A central hub for users to access all of MindWell's features, track their mood, and review their assessment history.
* **Dynamic and Responsive UI:** A clean, intuitive, and user-friendly interface built with HTML, CSS, and JavaScript for a seamless user experience across all devices.

---

## How It Works: A Multimodal Approach

MindWell's strength lies in its ability to fuse data from multiple sources to create a comprehensive picture of a user's mental health.

1.  **The Conversational Chatbot:** This model uses the Natural Language Toolkit (NLTK) for text preprocessing and a Sequential model built with TensorFlow/Keras. It's trained on the `merged_dataset_intents.json` file to recognize user intent and provide appropriate and supportive responses.

2.  **The Multimodal Mental Health Assessment Model:** This is where MindWell truly innovates. The deep learning model (`text_final_model.h5`) will be extended to become a multimodal fusion model. Here's the vision:
    * **Text Analysis:** The model will continue to process the user's answers from the `health.html` questionnaire, tokenizing and analyzing the text for sentiment and key themes.
    * **Audio Analysis (Future):** By integrating audio recording capabilities, MindWell will capture the user's voice as they answer questions. The audio will be processed to extract features like pitch, jitter, shimmer, and speaking rate. These features, which can be indicative of stress, depression, or anxiety, will be fed into the model.
    * **Visual Analysis (Future):** With user consent, MindWell will use the webcam to capture video during the assessment. Computer vision models will then analyze facial landmarks, eye movement, and micro-expressions to detect emotions that might not be explicitly stated.
    * **Data Fusion:** The features from text, audio, and video will be combined and fed into a sophisticated neural network. This "fusion" allows the model to learn the complex interplay between these different modalities, leading to a more accurate and holistic assessment of the user's mental state.

---

## Technologies Used

* **Backend:** Python, Flask, Flask-SQLAlchemy
* **Machine Learning & Deep Learning:** TensorFlow, Keras, Scikit-learn, NLTK
* **Frontend:** HTML, CSS, JavaScript
* **Database:** SQLite
* **Libraries:** NumPy, Pandas, Pickle

### Planned for Multimodality:

* **Audio Processing:** Libraries like Librosa for audio feature extraction.
* **Computer Vision:** OpenCV and facial recognition libraries (e.g., Dlib, OpenFace) for analyzing video feeds.
* **Real-time Communication:** WebRTC for streaming audio and video from the user's browser to the server.

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/adarsh-naik-2004/mindwell.git
    cd mindwell
    ```

2.  **Create and activate a virtual environment:**
    * **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```sh
    pip install Flask Flask-SQLAlchemy tensorflow scikit-learn nltk numpy pandas
    ```

4.  **Download NLTK data:**
    Run the Python interpreter and download the necessary NLTK packages.
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

### Running the Application

1.  **Run the Flask application:**
    ```sh
    python app.py
    ```

2.  **Open your browser:**
    Navigate to `http://127.0.0.1:5000` to view and interact with the application.

---

## Future Development: The Road to a Truly Multimodal Platform

The "mindwell" project is at an exciting stage. Here are the next steps to realize its full multimodal potential:

1.  **Audio Recording and Processing:**
    * Implement WebRTC to capture audio from the user's microphone in the `health.html` page.
    * On the server-side, use a library like Librosa to extract features from the audio stream in real-time.
    * Integrate these audio features into the deep learning model.

2.  **Video and Facial Expression Analysis:**
    * Use WebRTC to stream video from the user's webcam.
    * Employ OpenCV to process the video feed and detect faces.
    * Integrate a facial recognition library to extract facial action units (AUs) and recognize emotions.
    * Feed these visual features into the multimodal model.

3.  **Model Retraining and Fusion:**
    * The existing `text_final_model.h5` will need to be redesigned to accept and process the new audio and video features.
    * This will involve creating a "fusion" layer in the neural network that can effectively combine the information from the different modalities.
    * The model will then need to be retrained on a new, multimodal dataset.

4.  **Ethical Considerations:**
    * Implementing these features requires a strong commitment to user privacy and data security.
    * It will be crucial to be transparent with users about what data is being collected and how it is being used.
    * Obtaining explicit user consent before accessing their camera or microphone is non-negotiable.

---
