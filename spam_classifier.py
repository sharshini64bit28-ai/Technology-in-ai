import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def download_and_prepare_data():
    """Downloads and extracts the SMS Spam Collection dataset."""
    url = "https://uci.edu"
    zip_path = "sms_spam.zip"
    extract_path = "sms_data"
    
    if not os.path.exists(extract_path):
        print("📥 Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(zip_path)
        print("✅ Download and extraction complete.")
    
    # Load dataset into pandas dataframe
    data_file = os.path.join(extract_path, "SMSSpamCollection")
    df = pd.read_csv(data_file, sep='\t', names=['label', 'message'])
    return df

def main():
    print("=== PROJECT 1: SPAM EMAIL CLASSIFIER ===")
    
    # 1. Fetch data
    df = download_and_prepare_data()
    print(f"📊 Dataset Loaded: {df.shape[0]} rows.")
    
    # 2. Encode target labels (ham = 0, spam = 1)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label_num']
    
    # 3. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Convert text data into numerical feature vectors (Bag of Words)
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Train Multinomial Naive Bayes Classifier
    print("🤖 Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # 6. Evaluate Model
    y_pred = model.fit(X_train_vec, y_train).predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n📈 Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # 7. Live Demonstration / Testing
    print("\n🔮 Live Testing Demonstration:")
    test_emails = [
        "Congratulations! You've won a free $1000 Walmart gift card. Click here to claim your prize now!",
        "Hey, are we still meeting up for lunch today at 1 PM? Let me know.",
        "URGENT: Your account access has been restricted. Update your password immediately by clicking this link."
    ]
    
    test_vec = vectorizer.transform(test_emails)
    predictions = model.predict(test_vec)
    
    for email, pred in zip(test_emails, predictions):
        label = "🚨 SPAM" if pred == 1 else "✅ HAM (Legit)"
        print(f"Text: \"{email}\"\nPrediction: {label}\n")

if __name__ == "__main__":
    main()
