import os
import json
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import speech_recognition as sr
from flask import Flask, request, render_template, redirect, session, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from tensorflow.keras.saving import register_keras_serializable
from flask_mail import Mail, Message
from tensorflow.keras.utils import custom_object_scope


app = Flask(__name__)


mail = Mail(app)

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '' # your email id  
app.config['MAIL_PASSWORD'] = '' # password



app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test1.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'



lemmatizer = WordNetLemmatizer()
chatbot_model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('merged_dataset_intents.json'))



class CustomHubLayer(tf.keras.layers.Layer):
    def __init__(self, model_url, **kwargs):
        super(CustomHubLayer, self).__init__(**kwargs)
        self.hub_layer = hub.KerasLayer(model_url, output_shape=[512], input_shape=[], dtype=tf.string, trainable=True)

    def call(self, inputs):
        return self.hub_layer(inputs)

    def get_config(self):
        config = super(CustomHubLayer, self).get_config()
        config.update({"model_url": self.hub_layer.handle})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config["model_url"])

# Now, when loading the model, use the custom_object_scope
with custom_object_scope({'CustomHubLayer': CustomHubLayer}):
    model = load_model('text_final_model.h5')

# Now you can use the model for predictions
def predict_sentiment(text):
    preprocessed_text = tf.convert_to_tensor([text])
    logits = model.predict(preprocessed_text)
    probabilities = tf.nn.sigmoid(logits).numpy()
    score = np.mean(probabilities) * 100
    if score<1:
        score*=10
    return score



class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    emailwell = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name, emailwell):
        self.name = name
        self.email = email
        self.emailwell = emailwell
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = chatbot_model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understand. Could you please rephrase that?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result






def audio_model(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Audio Transcription: ", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        text = ''
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        text = ''
    return text


print([User.emailwell])


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        emailwell = request.form['emailwell']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password, emailwell=emailwell)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/home')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/home')
def home():
    if session.get('email'):
        user = User.query.filter_by(email=session['email']).first()

        return render_template('home.html', user=user)
    
    return redirect('/login')

@app.route('/health', methods=["GET", "POST"])
def health():
    if request.method == 'POST':
        feeling_text = request.form.get('feeling_text', '').strip()
        
        # Handle audio file if present
        audio_file = request.files.get('audio_file')
        if audio_file:
            feeling_text = audio_model(audio_file)
        
        # Get sentiment score
        if feeling_text:
            text_score = predict_sentiment(feeling_text)
            return redirect(url_for('result', text_score=text_score))
        
        # If no input provided, redirect back to health page
        return redirect(url_for('health'))
        
    return render_template('health.html')


@app.route('/result')
def result():
    text_score = request.args.get('text_score', default=0.0, type=float)
    if session.get('email'):
        
        if text_score < 40:
            send_low_score_email(User.name, User.emailwell)

    return render_template('result.html', text_score=text_score)

def send_low_score_email(user_name, well_wisher_email):
    """Send an email to the well-wisher if the test score is below 40%."""
    print({well_wisher_email})
    try:
        msg = Message(
            "Attention Needed: Low Score Alert",
            app.config['MAIL_USERNAME'],
            [well_wisher_email]
        )
        msg.body = f"Dear Well-wisher, \n\nPlease take care of your relation {user_name}. Their recent test score was below the required threshold.\n\nBest Regards,\nYour App Team"
        
        mail.send(msg)
        print(f"Email sent to {well_wisher_email} regarding {user_name}")
    
    except Exception as e:
        print(f"Failed to send email: {e}")


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    ints = predict_class(message)
    response = get_response(ints, intents)
    return jsonify({"response": response})

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)
