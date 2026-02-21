import random
import nltk

nltk.data.path.append('.')

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, get_suggestions, calculate_perplexity

# ----- CONFIG: Add your datasets here -----
DATASETS = {
    "Twitter": "./data/en_US.twitter.txt",
    "Shakespeare": "./data/merchant.txt",
    "Disney": "./data/disney.txt",
    "OnCampus": "./data/oncampus_no_numbers.txt"
}
TRAIN_SPLIT = 0.8
MINIMUM_FREQ = 2
K_SMOOTH = 1.0

# ----- TRAIN MODEL FOR EACH DATASET -----
model_data = {}
for ds_name, ds_path in DATASETS.items():
    with open(ds_path, "r", encoding="utf-8") as f:
        data = f.read()
    tokenized_data = get_tokenized_data(data)
    random.seed(87)
    random.shuffle(tokenized_data)
    train_size = int(len(tokenized_data) * TRAIN_SPLIT)
    train_data = tokenized_data[:train_size]
    test_data = tokenized_data[train_size:]
    train_data_processed, test_data_processed, vocabulary = preprocess_data(
        train_data, test_data, MINIMUM_FREQ
    )
    n_gram_counts_list = []
    for n in range(1, 5):  # 1-gram to 4-gram
        n_gram_counts = count_n_grams(train_data_processed, n)
        n_gram_counts_list.append(n_gram_counts)
    model_data[ds_name] = {
        "vocabulary": vocabulary,
        "n_gram_counts_list": n_gram_counts_list
    }

# ----- INTERACTIVE USER INPUT -----
while True:
    user_input = input("Enter a sentence (or 'q' to quit): ").lower().strip()
    if user_input == 'q':
        break   
    tokens = nltk.word_tokenize(user_input)
    print("=" * 60)
    print(f"Input: {user_input}")
    for ds_name, model in model_data.items():
        vocabulary = model["vocabulary"] 
        n_gram_counts_list = model["n_gram_counts_list"]
        
        # Safely get suggestions (defensive patch!)
        try:
            suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, K_SMOOTH)
            sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        except IndexError:
            sorted_suggestions = [("<NO DATA>", 0.0)]
    
        print(f"\n--- Dataset: {ds_name} ---")
        print("Suggestions:")
        for i, (word, prob) in enumerate(sorted_suggestions, 1):
            print(f"{i}. {word} (probability: {prob:.6f})")
        
        # Perplexity (with highest n-gram = 4-gram, so use n=3 and n=4 counts)
        try:
            vocabulary_size = len(vocabulary)
            perp = calculate_perplexity(
                tokens, n_gram_counts_list[3], n_gram_counts_list[3], vocabulary_size
            )
            print(f"Perplexity (4-gram): {perp:.2f}")
        except Exception:
            print("Perplexity (4-gram) not available.")
    print("=" * 60)
        
