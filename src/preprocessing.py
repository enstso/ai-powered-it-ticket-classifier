import re
import spacy
import spacy.lang
import pandas as pd
import nltk
from nltk import RegexpTokenizer, SnowballStemmer
from nltk.corpus import stopwords
from pathlib import Path


def preprocess_text_simple(texts):
    cleaned_texts = []
    for text in texts:
        text = tokenization(text)
        text = remove_stop_words(text)
        text = lemma_words(text)
        cleaned_texts.append(text)
    return cleaned_texts

def preprocess_text_nltk(texts):
    nltk.download("stopwords")
    french_stopwords = set(stopwords.words("french"))
    tokeninzer = RegexpTokenizer(r"[a-zA-ZÀ-ÿ]+")
    stemmer = SnowballStemmer("french")

    cleaned_texts = []

    for text in texts:
        text = str(text).lower()
        tokens = tokeninzer.tokenize(text)

        tokens = [
            token
            for token in tokens
            if token not in french_stopwords
        ]

        tokens = [
            stemmer.stem(token)
            for token in tokens
        ]

        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts


def remove_stop_words(text):
    text = " ".join(text)
    nlp = spacy.blank("fr")
    doc = nlp(text)

    new_tokens = [
        token.text
        for token in doc
        if not token.is_stop
    ]
    return " ".join(new_tokens)

def lemma_words(text):

    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text)
    exception_words = {
        "distant": "distant",
        "affiche": "affiche"
    }
    new_tokens = [
        exception_words.get(token.text.lower(), token.lemma_)
        for token in doc
    ]
    return " ".join(new_tokens)

def tokenization(text:str):
    return re.findall(r"[\w{1}]\w+",text.lower().strip())


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data/clean_tickets.csv"

    df = pd.read_csv(csv_path)

    print(preprocess_text_simple(df["text"]))
    print(preprocess_text_nltk(df["text"]))

