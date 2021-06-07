import time
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('keys/ideahub31-firebase-adminsdk-yl59k-f6da5b2634.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
model_paths = ['./models/Startup_Ideas_model', './models/AppIdeas_model', './models/gameideas_model']
tags = ['startup', 'app', 'game']
model_name = 'gpt2-medium' # gpt2, gpt2-medium, 
max_len = 80
num_return_sequences = 10

def game_idea_pipeline():
    idea_gen = pipeline('text-generation', model=model_path, tokenizer=model_name, config={'max_length':max_len})
    return idea_gen('<|title|>', max_length=max_len, num_return_sequences=num_return_sequences)

def game_idea_generate():
    # model_path = random.choice(model_paths)
    whichThing = 2
    tagPicked = tags[whichThing]
    model_path = model_paths[whichThing]
    model = TFGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("<|title|>", add_special_tokens=True, return_tensors='tf')

    generated_text_samples = model.generate( input_ids, 
        max_length=max_len, num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2, repetition_penalty=1.75,
        top_p=0.95, temperature=0.90, do_sample=True,
        top_k=125, early_stopping=True,
    )

    for i, beam in enumerate(generated_text_samples):
        decoded_idea = tokenizer.decode(beam, skip_special_tokens=True)
        firestore_push(decoded_idea[9:], tagPicked)
        print(f'{i}: {decoded_idea[9:]}\n')

def firestore_push(idea_text, tag_text):
    doc_ref = db.collection('generated').add({
        'displayName': 'GPT2-Bot', 
        'uid': '5aCGwn68JWUpOv3QSwduabVqqG62', #DFUxm8vnMgGKBh6AsZCcjXpek57H for GPT2
        'idea': idea_text, 
        'tag': tag_text, 
        'upvotes': 0, 
        'utc': int(time.time())
    })

if __name__ == '__main__':
    game_idea_generate()