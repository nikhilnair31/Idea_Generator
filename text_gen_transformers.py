import argparse
from transformers import pipeline, set_seed
set_seed(42)

def gen_text(in_text, max_length, num_return_sequences):
    generator = pipeline('text-generation', model='gpt2-medium')
    out_text = generator(in_text, max_length=max_length, num_return_sequences=num_return_sequences)
    print_text(out_text)

def print_text(out_text):
    print(f"\nOutput:\n")
    for dic in out_text:
        print(f"{ dic }\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='You can add a description here')
    parser.add_argument('-num','--num_return_sequences', help='num_return_sequences', type=int, required=True)
    parser.add_argument('-len','--max_length', help='max_length', type=int, required=True)
    parser.add_argument('-text','--text', help='text', type=str, required=True)
    args = parser.parse_args()

    gen_text(args.text, args.max_length, args.num_return_sequences)