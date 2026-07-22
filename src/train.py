import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_text_simple, preprocess_text_nltk

def baseline(x_training, y_training, x_testing, y_testing):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_training, y_training)
    y_baseline_pred = baseline.predict(x_testing)
    baseline_accuracy = accuracy_score(y_testing, y_baseline_pred)
    print(f"Baseline accuracy: {baseline_accuracy:.2f}") #0.25
    print(f"category most_frequent {baseline.classes_[baseline.class_prior_.argmax()]}")


def model_a_pipeline():
    return Pipeline([
        ("preprocessing",FunctionTransformer(preprocess_text_simple,validate=False)),
        ("tfidf",TfidfVectorizer()),
        ("classifier",
         LogisticRegression(
             max_iter=1000,
             random_state=42
        )
    )
    ])

def model_b_pipeline():
    return Pipeline([
        ("preprocessing",FunctionTransformer(preprocess_text_nltk,validate=False)),
        ("tfidf",TfidfVectorizer()),
        (
            "classifer",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data/clean_tickets.csv"

    df = pd.read_csv(csv_path)

    X = df["text"]
    y = df["category"]

    print("Nb total rows :",len(y))
    print("Nb of categories:",y.nunique())
    print(y.value_counts())

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    baseline(X_train,y_train, X_test, y_test)

