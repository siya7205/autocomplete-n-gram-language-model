import random
import nltk

nltk.data.path.append('.')

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, get_suggestions, calculate_perplexity

# Step 1 - Load and Preprocess Data
with open("./data/disney.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[:train_size]
test_data = tokenized_data[train_size:]

minimum_freq = 2  # You can change this!
train_data_processed, test_data_processed, vocabulary = preprocess_data(
    train_data, test_data, minimum_freq
)

# Step 2 - Train n-gram models (collect up to 5-gram for 4-gram perplexity)
n_gram_counts_list = []
max_n = 6
for n in range(1, max_n + 1):  # 1-gram through 5-gram
    n_gram_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_gram_counts)  
            
def get_user_input_suggestions(vocabulary, n_gram_counts_list, k=0):
    model_names = ["Unigram", "Bigram", "Trigram", "4-gram"]
    max_model_used = 4  # Up to 4-gram

    while True:
        user_input = input("Enter a sentence (or 'q' to quit): ").lower().strip()
        if user_input == 'q':
            break

        # Tokenize the user input
        tokens = nltk.word_tokenize(user_input)

        # Get suggestions (top-1 from each model)
        suggestions = get_suggestions(tokens, n_gram_counts_list[:max_model_used], vocabulary, k)
        sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        print("\nSuggestions by model:")
        for model_idx, (word, prob) in enumerate(sorted_suggestions[:max_model_used]):
            print(f"{model_names[model_idx]} suggests: {word} (probability: {prob:.6f})")
        print()
 # Perplexity for all n-gram orders
        print("Perplexity of your input:")
        for n in range(1, max_model_used + 1):  # n=1 you need n=2 counts, n=4 you need n=5 counts
            try:
                n_gram_counts = n_gram_counts_list[n-1]
                n_plus1_gram_counts = n_gram_counts_list[n]
                vocabulary_size = len(vocabulary)
                perp = calculate_perplexity(tokens, n_gram_counts, n_plus1_gram_counts, vocabulary_size)
                print(f"{n}-gram: {perp:.2f}")
            except Exception as e:
                print(f"{n}-gram: Perplexity not available (input too short or insufficient model).")
        print()
    
# Call the function interactively
get_user_input_suggestions(vocabulary, n_gram_counts_list)
        
        
