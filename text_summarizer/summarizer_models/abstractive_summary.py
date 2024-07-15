import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def sentence_case(text):
    if not text:
        return ''
    return text[0].upper() + text[1:].lower()

def sentence_case_large_text(text):
    sentences = sent_tokenize(text)
    sentences = [sentence_case(sentence) for sentence in sentences]
    return ' '.join(sentences)

def abstractive_summarization(text, model_name='t5-small', max_length=150, min_length=100, length_penalty=2.0, num_beams=4):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    preprocess_text = text.strip().replace("\n", " ")
    t5_input_text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(tokenized_text, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    final_summary = sentence_case_large_text(summary)
    return final_summary

