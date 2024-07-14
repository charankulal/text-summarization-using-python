import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, BertModel
import torch
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

text = """
Fish are aquatic animals that are typically cold-blooded and have gills throughout their lives.
They are found in both freshwater and saltwater environments, ranging from small ponds to the deep ocean.
Fish have a wide variety of shapes, sizes, and colors, adapted to their diverse habitats.
They play a crucial role in aquatic ecosystems, serving as both predators and prey.
Fish are also important to humans for food, recreational fishing, and as pets.
They have a complex range of behaviors and biological adaptations, such as schooling, camouflage, and specialized breeding techniques, making them a fascinating subject of study in marine biology.
"""

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

def compute_tfidf_scores(sentences):
    """
    Compute the TF-IDF scores for the sentences.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    return tfidf_matrix

def cosine_sim(matrix):
    """
    Compute the cosine similarity matrix.
    """
    return cosine_similarity(matrix, matrix)

def summarize_text(text, summary_ratio=0.3):
    """
    Summarizes the input text by selecting the most important sentences.

    Parameters:
    text (str): The text to summarize.
    summary_ratio (float): The ratio of sentences to include in the summary.

    Returns:
    str: The summarized text.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Tokenize and create word frequency dictionary
    word_frequency = {}
    pos_weight = {"NOUN": 2, "VERB": 1.5, "ADJ": 1, "ADV": 1}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.pos_ in pos_weight:
                word_frequency[word.text] = word_frequency.get(word.text, 0) + pos_weight[word.pos_]

    max_frequency = max(word_frequency.values())
    word_frequency = {word: freq / max_frequency for word, freq in word_frequency.items()}

    # Convert sentences generator to a list
    sentences = list(doc.sents)
    sentences_text = [sent.text for sent in sentences]

    # Score sentences based on word frequencies
    sent_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text in word_frequency:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_frequency[word.text]

    # Compute TF-IDF scores
    tfidf_matrix = compute_tfidf_scores(sentences_text)
    cosine_matrix = cosine_sim(tfidf_matrix)
    
    # Adjust sentence scores based on TF-IDF and cosine similarity
    for i, sent in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            if cosine_matrix[i][j] > 0.3:  # Penalize similar sentences more heavily
                sent_scores[sent] -= cosine_matrix[i][j]

    # Incorporate BERT embeddings for better sentence ranking
    sentence_embeddings = [bert_embedding(sent.text) for sent in sentences]
    bert_cosine_matrix = cosine_similarity(sentence_embeddings)

    # Adjust sentence scores based on BERT cosine similarity
    for i, sent in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            if bert_cosine_matrix[i][j] > 0.3:  # Penalize similar sentences more heavily
                sent_scores[sent] -= bert_cosine_matrix[i][j]

    # Select top sentences
    select_len = int(len(sentences) * summary_ratio)
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    return " ".join([sent.text for sent in summary])

# Summarize the text
summary = summarize_text(text)

print("Original text \n", text)
print("\n\nSummary \n", summary)
print("Length of original text:", len(text.split()), "words")
print("Length of Summary text:", len(summary.split()), "words")
