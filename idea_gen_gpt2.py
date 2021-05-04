from transformers import pipeline

chef = pipeline('text-generation', model='./gpt2', tokenizer='gpt2',config={'max_length':128})

result = chef('Zuerst Tomaten')[0]['generated_text']