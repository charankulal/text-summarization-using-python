import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """
Fish are aquatic animals that are typically cold-blooded and have gills throughout their lives.
They are found in both freshwater and saltwater environments, ranging from small ponds to the deep ocean.
Fish have a wide variety of shapes, sizes, and colors, adapted to their diverse habitats.
They play a crucial role in aquatic ecosystems, serving as both predators and prey.
Fish are also important to humans for food, recreational fishing, and as pets.
Fish have a complex range of behaviors and biological adaptations, such as schooling, camouflage, and specialized breeding techniques, making them a fascinating subject of study in marine biology.
"""

stopwords = list(STOP_WORDS)


nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

tokens = [token.text for token in doc]

word_frequency = {}

for word in doc:
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        if word.text not in word_frequency.keys():
            word_frequency[word.text] = 1
        else:
            word_frequency[word.text] += 1

max_frequency = max(word_frequency.values())

for word in word_frequency.keys():
    word_frequency[word] = word_frequency[word] / max_frequency

sent_tokens = [sent for sent in doc.sents]

sent_scores = {}

for sent in sent_tokens:
    for word in sent:
        if word.text in word_frequency.keys():
            if sent not in sent_scores.keys():
                sent_scores[sent] = word_frequency[word.text]
            else:
                sent_scores[sent] += word_frequency[word.text]

select_len = int(len(sent_tokens) * 0.3)

summary = nlargest(select_len, sent_scores, key=sent_scores.get)

summary_str = [word.text for word in summary]
summary = " ".join(summary_str)

print("Original text \n", text)
print("\n\nSummary \n", summary)
print("Length of original text ", len(text.split(" ")), " words")
print("Length of Summary text ", len(summary.split(" ")), " words")
