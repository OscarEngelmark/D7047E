import pandas as pd
import nltk
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('stopwords',    quiet=True)


def _preprocess_chunk(sentences: pd.Series) -> pd.Series:
    """Preprocess one chunk of sentences. Module-level so multiprocessing can pickle it."""
    stop_words = set(stopwords.words('english'))
    s = sentences.str.lower()
    s = s.replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    s = s.replace(r'(?:\d{1,3}\.){3}\d{1,3}', '', regex=True)
    s = s.str.replace(r'[^\w\s]', '', regex=True)
    s = s.replace(r'\d+', '', regex=True)
    s = s.apply(lambda text: ' '.join(w for w in word_tokenize(text) if w not in stop_words))
    return s


def preprocess_pandas(data: pd.DataFrame) -> pd.DataFrame:
    """Clean review text: lowercase, remove emails/IPs/punctuation/digits/stopwords."""
    df = data.copy()
    n_workers = cpu_count() or 1
    n = len(df)
    size = max(1, (n + n_workers - 1) // n_workers)
    chunks = [df['Sentence'].iloc[i:i + size] for i in range(0, n, size)]
    with Pool(processes=n_workers) as pool:
        results = pool.map(_preprocess_chunk, chunks)
    df['Sentence'] = pd.concat(results)
    return df
