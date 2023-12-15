import re
import string
from torchtext.data import get_tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = get_tokenizer('basic_english')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


def remove_alphanums(s):
    for word in s.split():
        if word.strip(string.punctuation).isdigit():
            yield word
        else:
            yield ''.join(char for char in word if not char.isdigit())


def preprocess_text_lstm(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove digits attached to words
    text = ' '.join(remove_alphanums(text))
    extended_punctuation = string.punctuation + "‘’"
    # Remove punctuation
    text = ''.join([char for char in text if char not in extended_punctuation])
    # Tokenize
    words = tokenizer(text)
    # Remove stopwords and lemmatize
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(lemmatized_words)
