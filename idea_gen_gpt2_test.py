import json
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

idea_list = []
model_path = './models'
model_name = 'gpt2'

def get_idea(max_len):
    idea_gen = pipeline('text-generation', model=model_path, tokenizer=model_name, config={'max_length':128})
    return idea_gen('<|title|>', max_length=max_len, num_return_sequences=1)

def game_idea(max_len):
    model = TFGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("<|title|>", add_special_tokens=True, return_tensors='tf')

    generated_text_samples = model.generate( input_ids, 
        max_length=max_len, num_return_sequences=5,
        no_repeat_ngram_size=2, repetition_penalty=1.75,
        top_p=0.95, temperature=0.90, do_sample=True,
        top_k=125, early_stopping=True,
    )

    for i, beam in enumerate(generated_text_samples):
        decoded_idea = tokenizer.decode(beam, skip_special_tokens=True)
        idea_list.append({"id":i, "decoded_idea":decoded_idea})
        print(f'{i}: {decoded_idea}\n')

if __name__ == '__main__':
    game_idea(300)