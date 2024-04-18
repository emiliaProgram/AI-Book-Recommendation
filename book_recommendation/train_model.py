import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_and_save_model():
    data = pd.read_csv('data.csv', header=None)
    inputs = data[0].tolist()
    labels = data[1].tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.25, random_state=42)

    # Vectorize data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2, max_df=0.5)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == '__main__':
    train_and_save_model()
