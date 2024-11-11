from flask import Flask, request, jsonify, render_template
import joblib

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
