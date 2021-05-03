from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
train_path = './data/reddit/txt/reddit_gameideas_train_tot.txt'
test_path = './data/reddit/txt/reddit_gameideas_test_tot.txt'

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset( tokenizer=tokenizer, file_path=train_path, block_size=128)
    test_dataset = TextDataset( tokenizer=tokenizer, file_path=test_path, block_size=128)   
    data_collator = DataCollatorForLanguageModeling( tokenizer = tokenizer, mlm = False, )
    return train_dataset,test_dataset,data_collator

train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

training_args = TrainingArguments(
    output_dir="./gpt2_model", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=2, # batch size for training
    per_device_eval_batch_size=2,  # batch size for evaluation
    eval_steps = 10, # Number of update steps between two evaluations.
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

trainer.train()
trainer.save_model()

from transformers import pipeline
chef = pipeline('text-generation', model='./gpt2', tokenizer='gpt2',config={'max_length':128})

result = chef('Zuerst Tomaten')[0]['generated_text']