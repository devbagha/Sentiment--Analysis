from application import app
from flask import render_template, request
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained SVM model
with open('model.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

# Load the saved vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    try:
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        words = word_tokenize(text)  # Tokenize text
        words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
        return ' '.join(words)
    except:
        return ""

def analyze_sentiment(review, vectorizer):
    preprocessed_input = preprocess_text(review)

    # Transform the preprocessed input text using the loaded vectorizer
    input_vec = vectorizer.transform([preprocessed_input])

    # Make sentiment prediction using the loaded SVM model
    predicted_sentiment = svc_model.predict(input_vec)[0]
    return predicted_sentiment

@app.route("/", methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
def index():
    sentiment_result = None
    review = None

    if request.method == 'POST':
        review = request.form['review']
        sentiment_result = analyze_sentiment(review, loaded_vectorizer)

    return render_template('index.html',  sentiment_result=sentiment_result, review=review)
