from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Завантажте модель та векторизатор
model = joblib.load('fake_news_detector.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    result = 'Fake' if prediction[0] == 0 else 'Real'
    return jsonify({'prediction': result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)