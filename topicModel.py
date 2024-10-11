import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')


# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans('', '', string.punctuation)
    tokens = text.lower().translate(translator).split()
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


# Function to perform topic modeling
def topic_modeling(text_data, n_topics=5, n_top_words=10, method='lda'):
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(text_data)

    if method == 'lda':
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    elif method == 'nmf':
        model = NMF(n_components=n_topics, random_state=42)
    else:
        raise ValueError("Method must be 'lda' or 'nmf'")

    model.fit(dtm)

    # Get the words for each topic
    feature_names = vectorizer.get_feature_names_out()
    for index, topic in enumerate(model.components_):
        print(f"Topic {index + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[-n_top_words:]]))
        print("\n")


# Main function
def main():
    print("Enter your text (end with an empty line):")
    user_input = []
    while True:
        line = input()
        if line.strip() == "":
            break
        user_input.append(line)

    # Preprocess the input text
    processed_text = [preprocess_text(" ".join(user_input))]

    # Perform topic modeling
    topic_modeling(processed_text, n_topics=3, n_top_words=5, method='lda')


if __name__ == "__main__":
    main()
