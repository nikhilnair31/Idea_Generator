import json
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, jsonify, request

app = Flask(__name__)

def get_idea(input_text, max_len):
    idea_gen = pipeline('text-generation', model='./trained_model', tokenizer='gpt2-medium', config={'max_length':128})
    return idea_gen(input_text, max_length=max_len, num_return_sequences=1)

def game_idea(max_len):
    idea_list = []
    model = TFGPT2LMHeadModel.from_pretrained("./trained_model", from_pt=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
    return idea_list

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        max_len = request.json["len"]
        final_idea = game_idea(int(max_len))
        return json.dumps(final_idea)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True)