import json, os

result = []
text_info = {}
tag_info = {}

task1_answer = "./audio_folder/task1_answer.txt"
task2_answer = "./audio_folder/task2_answer.txt"

# tag list
with open("./NER_Shi/tag.txt", 'r', encoding='utf-8') as file:
    tag_list = file.readlines()
    tag_list = [tag.strip() for tag in tag_list]

# text info
with open(task1_answer, 'r', encoding='utf-8') as file:
    task1_data = file.readlines()
for line in task1_data:
    parts = line.strip().split("\t")
    filename = parts[0]
    transcription = parts[1]
    if filename not in text_info:
        text_info[filename] = transcription

# tag info
with open(task2_answer, 'r') as file:
    task2_data = file.readlines()
for line in task2_data:
    parts = line.strip().split("\t")
    filename = parts[0]
    tag = parts[1]
    word = parts[4]
    if filename not in tag_info:
        tag_info[filename] = []
    tag_info[filename].append({
        "tag": tag,
        "word": word
    })

adjusted_tag_list = ["O"]
for tag in tag_list:
    adjusted_tag_list.append(f"B-{tag}")
    adjusted_tag_list.append(f"I-{tag}")

for filename, transcription in text_info.items():
    if filename not in tag_info:
        continue
    tokens = transcription.split()
    tokens = [token.strip('.,!?"') for token in tokens if token.strip('.,!?"')]
    ner_tags = []
    tags = tag_info[filename].copy()
    tag_idx = 0
    for token in tokens:
        if tag_idx < len(tags) and token == tags[tag_idx]["word"]:
            tag = tags[tag_idx]["tag"]
            if tag not in tag_list:
                tag_idx += 1
                continue
            tag_num = adjusted_tag_list.index(f"B-{tag}") if tag_idx == 0 else adjusted_tag_list.index(f"I-{tag}")
            ner_tags.append(tag_num)
            tag_idx += 1
        else:
            ner_tags.append(0)
    
    if len(tokens) != len(ner_tags):
        print(f"Warning: Length mismatch for {filename}. Tokens: {len(tokens)}, NER Tags: {len(ner_tags)}")
        continue
    
    result.append({
        "tokens": tokens,
        "ner_tags": ner_tags,
    })

output_file = "./NER_Shi/combined_data.json"
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False, indent=4)
