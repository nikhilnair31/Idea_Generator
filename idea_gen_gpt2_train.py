from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

train_path = './data/txt/reddit_gameideas_20000_1619827200_1462060800_train_tot.txt'
test_path = './data/txt/reddit_gameideas_20000_1619827200_1462060800_test_tot.txt'
output_path = "./models"

def trainsave(trainer):
    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataset = TextDataset( tokenizer=tokenizer, file_path=train_path, block_size=256)
    test_dataset = TextDataset( tokenizer=tokenizer, file_path=test_path, block_size=256)   
    data_collator = DataCollatorForLanguageModeling( tokenizer = tokenizer, mlm = False, )
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=output_path, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        prediction_loss_only=True,
        per_device_train_batch_size=4, # batch size for training
        per_device_eval_batch_size=4,  # batch size for evaluation
        num_train_epochs=3, # number of training epochs
        eval_steps = 200, # Number of update steps between two evaluations.
        save_steps=400, # after # steps model is saved
        warmup_steps=250,# number of warmup steps for learning rate scheduler
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainsave(trainer)