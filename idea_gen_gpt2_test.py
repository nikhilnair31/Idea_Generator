# setup imports to use the model
# from transformers import pipeline
# train_path = './data/reddit/txt/reddit_gameideas_train_tot.txt'
# test_path = './data/reddit/txt/reddit_gameideas_test_tot.txt'
# output_path = "./trained_model"
# def print_clean(idea_output):
#     for idea in idea_output:
#         print(f'\n{idea["generated_text"]}\n')
# idea_gen = pipeline('text-generation',model=output_path, tokenizer='gpt2',config={'max_length':300})
# idea_output = idea_gen('<|title|>', max_length=300, num_return_sequences=5)
# print_clean(idea_output)

# setup imports to use the model
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
output_path = "./trained_model"
model = TFGPT2LMHeadModel.from_pretrained(output_path, from_pt=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_ids = tokenizer.encode("<|title|>", add_special_tokens=True, return_tensors='tf')
generated_text_samples = model.generate( input_ids, 
    max_length=300, num_return_sequences=5,
    no_repeat_ngram_size=2, repetition_penalty=1.75,
    top_p=0.95, temperature=0.90,
    do_sample=True, top_k=125,
    early_stopping=True
)
for i, beam in enumerate(generated_text_samples):
    print("{}: {}\n".format(i,tokenizer.decode(beam, skip_special_tokens=True)))