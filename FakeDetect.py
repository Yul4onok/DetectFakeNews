import tkinter as tk 
import joblib
import re
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load('fake_news_detector.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
lemmatizer = WordNetLemmatizer()

# Preprocess function
def preprocess_text(text):
    # Remove URLs, hashtags, numbers, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Predict function
def predict_fake_news():
    content = input_text.get("1.0", tk.END)
    processed_content = preprocess_text(content)
    vectorized_content = vectorizer.transform([processed_content])
    prediction = model.predict(vectorized_content)
    result = "Fake News" if prediction[0] == 0 else "Real News"
    result_label.config(text=result)

# Function to paste from clipboard
def paste_from_clipboard():
    input_text.insert(tk.INSERT, app.clipboard_get())

# Tkinter GUI setup
app = tk.Tk()
app.title("Fake News Detector")
app.geometry("400x300")

# Create menu bar with paste option
menu_bar = tk.Menu(app)
edit_menu = tk.Menu(menu_bar, tearoff=0)
edit_menu.add_command(label="Paste", command=paste_from_clipboard)
menu_bar.add_cascade(label="Edit", menu=edit_menu)
app.config(menu=menu_bar)

# Input field
input_text = tk.Text(app, height=10, width=50)
input_text.pack()

# Predict button
predict_button = tk.Button(app, text="Analyze News", command=predict_fake_news)
predict_button.pack()

# Result display
result_label = tk.Label(app, text="", font=("Helvetica", 16))
result_label.pack()

app.mainloop()
