import sys
import re
from collections import Counter
from textblob import TextBlob
import spacy
import matplotlib.pyplot as plt

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """Preprocess the text by converting to lowercase and removing non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def term_frequency_analysis(text):
    """Perform term frequency analysis on the text."""
    words = text.split()
    word_count = Counter(words)
    return word_count


def sentiment_analysis(text):
    """Perform sentiment analysis on the text."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity


def named_entity_recognition(text):
    """Perform named entity recognition on the text."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities


def plot_term_frequencies(term_frequencies):
    """Plot the term frequencies."""
    terms, freqs = zip(*term_frequencies.most_common(10))
    plt.figure(figsize=(10, 6))
    plt.bar(terms, freqs)
    plt.xlabel('Terms')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Terms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python transcription_analysis.py <transcript_file>")
        sys.exit(1)

    transcript_file = sys.argv[1]

    try:
        with open(transcript_file, 'r') as file:
            transcript_text = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{transcript_file}' was not found.")
        sys.exit(1)

    print("Analyzing transcript...")

    # Preprocess the text
    processed_text = preprocess_text(transcript_text)

    # Perform term frequency analysis
    term_frequencies = term_frequency_analysis(processed_text)

    # Perform sentiment analysis
    polarity, subjectivity = sentiment_analysis(transcript_text)

    # Perform named entity recognition
    entities = named_entity_recognition(transcript_text)

    # Display the results
    print("\nTerm Frequency Analysis:")
    for term, freq in term_frequencies.most_common(10):
        print(f"{term}: {freq}")

    print("\nSentiment Analysis:")
    print(f"Polarity: {polarity:.2f} (Range: -1 to 1)")
    print(f"Subjectivity: {subjectivity:.2f} (Range: 0 to 1)")

    print("\nNamed Entities:")
    print(", ".join(set(entities)))

    # Plot term frequencies
    plot_term_frequencies(term_frequencies)


if __name__ == "__main__":
    main()
