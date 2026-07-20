def preprocess_text(text:str)->str:
    lower_text = text.lower()
    tokenization = lower_text.split()
    print(tokenization)

if __name__ == "__main__":
    preproces_text("Les chats mangent vite.")