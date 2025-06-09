from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model = AutoModelForTokenClassification.from_pretrained("./model/ner_shi_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./model/ner_shi_model", local_files_only=True)

ner_pipeline = pipeline(
    task="token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)