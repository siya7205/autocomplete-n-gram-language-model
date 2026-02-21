import random
import nltk
from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, calculate_perplexity

# Utility for reproducibility
random.seed(87)
nltk.download('punkt')

DATASETS = {
    "Twitter": "./data/en_US.twitter.txt",
    "Disney": "./data/disney.txt"
}

def load_and_split_dataset(path, min_freq=2):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    tokenized_data = get_tokenized_data(data)
    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[:train_size]
    test_data = tokenized_data[train_size:]
    train_processed, test_processed, vocab = preprocess_data(train_data, test_data, min_freq)
    return train_processed, test_processed, vocab

def train_ngram_model(train_data, max_n=5):
    ngram_counts_list = []
    for n in range(1, max_n+1):
        ngram_counts = count_n_grams(train_data, n)
        ngram_counts_list.append(ngram_counts)
    return ngram_counts_list

def average_perplexity(test_data, n, ngram_counts, n_plus1_gram_counts, vocab):
    perps = []
    for tokens in test_data:
        # only evaluate sentences of at least n tokens
        if len(tokens) >= n:
            try:
                perp = calculate_perplexity(tokens, ngram_counts, n_plus1_gram_counts, len(vocab))
                perps.append(perp)
            except:
                continue
    return sum(perps) / len(perps) if perps else float('nan')

datasets_processed = {}
for name, path in DATASETS.items():
    datasets_processed[name] = load_and_split_dataset(path, min_freq=2)

# --- Cross-evaluation loop ---
for train_name in DATASETS:
    train_data, _, train_vocab = datasets_processed[train_name]
    train_ngram_counts = train_ngram_model(train_data, max_n=5)
    
    for test_name in DATASETS:
        _, test_data, test_vocab = datasets_processed[test_name]
        print(f"\nTrain on: {train_name} | Test on: {test_name}")
        for n in range(1, 5):
            ngram_counts = train_ngram_counts[n-1]
            n_plus1_gram_counts = train_ngram_counts[n]
            avg_perp = average_perplexity(test_data, n, ngram_counts, n_plus1_gram_counts, train_vocab)
            print(f"  {n}-gram average perplexity = {avg_perp:.2f}")


    
    
        
