import torch
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
train_path = './data/reddit/txt/reddit_gameideas_train_tot.txt'
test_path = './data/reddit/txt/reddit_gameideas_test_tot.txt'
output_path = "./trained_model"

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset( tokenizer=tokenizer, file_path=train_path, block_size=128)
    test_dataset = TextDataset( tokenizer=tokenizer, file_path=test_path, block_size=128)   
    data_collator = DataCollatorForLanguageModeling( tokenizer = tokenizer, mlm = False, )
    return train_dataset,test_dataset,data_collator

def print_clean(idea_output):
    for idea in idea_output:
        print(f'\n{idea["generated_text"]}\n')

train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir= output_path, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=5, # number of training epochs
    per_device_train_batch_size=4, # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    eval_steps = 200, # Number of update steps between two evaluations.
    save_steps=400, # after # steps model is saved
    warmup_steps=250,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

trainer.train()
trainer.save_model()

idea_gen = pipeline('text-generation',model=output_path, tokenizer='gpt2',config={'max_length':128})

idea_output = idea_gen('A platformer where', max_length=96, num_return_sequences=3)
print_clean(idea_output)