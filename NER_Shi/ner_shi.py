from NER_Shi import ner_pipeline

def predict(text):
    if not text:
        return None
    predictions = ner_pipeline(text)
    print(predictions)