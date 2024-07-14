import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

text = """
Fish are aquatic animals that are typically cold-blooded and have gills throughout their lives.
They are found in both freshwater and saltwater environments, ranging from small ponds to the deep ocean.
Fish have a wide variety of shapes, sizes, and colors, adapted to their diverse habitats.
They play a crucial role in aquatic ecosystems, serving as both predators and prey.
Fish are also important to humans for food, recreational fishing, and as pets.
Fish have a complex range of behaviors and biological adaptations, such as schooling, camouflage, and specialized breeding techniques, making them a fascinating subject of study in marine biology.
"""

def abstractive_summarization(text, model_name='t5-small', max_length=150, min_length=40, length_penalty=2.0, num_beams=4):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    preprocess_text = text.strip().replace("\n", " ")
    t5_input_text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(tokenized_text, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


summary = abstractive_summarization(text)

print("Original text \n", text)
print("\n\nSummary \n", summary)



print("Length of original text:", len(text.split()), "words")
print("Length of Summary text:", len(summary.split()), "words")
