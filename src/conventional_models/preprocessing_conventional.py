import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK datasets
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

english_stopwords = set(stopwords.words("english"))


def preprocess_text_conventional(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation + "‘’"])

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    # tokens = [token for token in tokens if token not in english_stopwords]

    # Tokenization
    return ' '.join(tokens)
