import tensorflow as tf
from transformers import pipeline, set_seed
set_seed(42)

def gen_text(in_text, max_length, num_return_sequences):
    generator = pipeline('text-generation', model='gpt2-medium')
    out_text = generator(in_text, max_length=40, num_return_sequences=5)
    print_text(out_text)

def print_text(out_text):
    print(f"\nOutput:\n")
    for dic in out_text:
        print(f"{ dic }\n")

if __name__=="__main__":
    num_return_sequences = 5
    max_length = 40
    in_text = "Make a startup"
    gen_text(in_text, max_length, num_return_sequences)