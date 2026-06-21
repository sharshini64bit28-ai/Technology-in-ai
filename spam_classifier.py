import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def download_and_prepare_data():
    """Loads a sample of the SMS Spam dataset entirely offline without any network calls."""
    print("📦 Loading offline dataset directly from script memory...")
    
    # Raw tab-separated dataset content built directly into the file
    raw_data = """ham	Hey, are we still meeting up for lunch today at 1 PM? Let me know.
ham	Can you pick up some milk on your way home? Thanks!
ham	Yeah everything is fine over here. Call you later.
ham	Are you coming to the party tonight or staying in?
ham	Just finished my assignments, let's grab a coffee.
spam	Congratulations! You've won a free $1000 Walmart gift card. Click here to claim your prize now!
spam	URGENT: Your account access has been restricted. Update your password immediately by clicking this link.
spam	WINNER! As a valued network customer you have been selected to receive a £900 prize! Call 09061701461 claim code KL341.
spam	FREE ringtone! Reply with 'REAL' to get yours now. Special limited offer.
spam	Private Account Statement for your eyes only. Claim your hidden cash bonus within 24 hours."""
    
    # Read the text variable as if it were an extracted file
    df = pd.read_csv(io.StringIO(raw_data), sep='\t', names=['label', 'message'])
    print("✅ Dataset loaded successfully.")
    return df

def main():
    print("=== PROJECT 1: SPAM EMAIL CLASSIFIER ===")
    
    df = download_and_prepare_data()
    print(f"📊 Dataset Loaded: {df.shape[0]} rows.")
    
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label_num']
    
    # Using a small test split optimized for the offline baseline dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("🤖 Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n📈 Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
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
