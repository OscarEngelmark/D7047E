import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('stopwords',    quiet=True)

def preprocess_pandas(data: pd.DataFrame) -> pd.DataFrame:
    """Clean review text: lowercase, remove emails/IPs/punctuation/digits/stopwords."""
    df = data.copy()
    df['Sentence'] = df['Sentence'].str.lower()
    df['Sentence'] = df['Sentence'].replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)    # emails
    df['Sentence'] = df['Sentence'].replace(r'(?:\d{1,3}\.){3}\d{1,3}', '', regex=True)            # IP addresses
    df['Sentence'] = df['Sentence'].str.replace(r'[^\w\s]', '', regex=True)                        # punctuation
    df['Sentence'] = df['Sentence'].replace(r'\d+', '', regex=True)                                # digits

    stop_words = set(stopwords.words('english'))
    df['Sentence'] = df['Sentence'].apply(
        lambda text: ' '.join(
            w for w in word_tokenize(text) if w not in stop_words
        )
    )
    return df
