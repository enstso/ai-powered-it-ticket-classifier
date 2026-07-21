import re
import spacy
import spacy.lang
def preprocess_text(text:str)->str:
    tokens = tokenization(text)
    print(tokens)
    tokens = remove_stop_words(tokens)
    print(tokens)
    tokens = lemma_words(tokens)
    print(tokens)

def remove_stop_words(tokens:list[str])->list[str]:
    text = " ".join(tokens)
    nlp = spacy.blank("fr")
    doc = nlp(text)

    new_tokens = [
        token.text
        for token in doc
        if not token.is_stop
    ]

    return new_tokens

def lemma_words(tokens:list[str])->list[str]:
    text = " ".join(tokens)
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text)

    new_tokens = [
        token.lemma_
        for token in doc
    ]
    return new_tokens

def tokenization(text:str)->str:
    return re.findall(r"\w+",text.lower().strip())

if __name__ == "__main__":
    preprocess_text("Les chats mangent vite.")