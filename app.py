from transformers import pipeline
from flask import Flask, jsonify, request

app = Flask(__name__)

def get_idea(input_text, max_len):
    idea_gen = pipeline('text-generation', model='./trained_model', tokenizer='gpt2-medium', config={'max_length':128})
    return idea_gen(input_text, max_length=max_len, num_return_sequences=1)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.json["text"]
        max_len = request.json["len"]
        final_idea = get_idea(str(input_text), int(max_len))
        return jsonify({'idea_text': final_idea[0]["generated_text"]})

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True)