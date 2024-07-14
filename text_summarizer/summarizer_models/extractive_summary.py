import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


nltk.download('punkt')
nltk.download('stopwords')


def compute_tfidf_scores(sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    return tfidf_matrix

def cosine_sim(matrix):
    return cosine_similarity(matrix, matrix)

def extractive_summary_func(text, summary_ratio=0.3):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    word_frequency = {}
    pos_weight = {"NOUN": 2, "VERB": 1.5, "ADJ": 1, "ADV": 1}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.pos_ in pos_weight:
                word_frequency[word.text] = word_frequency.get(word.text, 0) + pos_weight[word.pos_]

    max_frequency = max(word_frequency.values())
    word_frequency = {word: freq / max_frequency for word, freq in word_frequency.items()}

    sentences = list(doc.sents)
    sentences_text = [sent.text for sent in sentences]

    sent_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text in word_frequency:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_frequency[word.text]

    tfidf_matrix = compute_tfidf_scores(sentences_text)
    cosine_matrix = cosine_sim(tfidf_matrix)
    
    for i, sent in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            if cosine_matrix[i][j] > 0.3:  # Penalize similar sentences more heavily
                sent_scores[sent] -= cosine_matrix[i][j]

    select_len = int(len(sentences) * summary_ratio)
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    return " ".join([sent.text for sent in summary])

