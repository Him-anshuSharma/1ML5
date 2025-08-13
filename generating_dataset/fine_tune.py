import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

# %%
import transformers
print(transformers.__version__)
print(transformers.TrainingArguments)


# %%
df = pd.read_csv('diary_analysis.csv')

emotion_cols = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

dataset = Dataset.from_pandas(df)

model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(emotion_cols),
    problem_type="regression"  # multi-label regression
)

# Step 5: Tokenize the text (journals)
def preprocess_function(examples):
    return tokenizer(examples["journal"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(preprocess_function, batched=True)

# Step 6: Attach labels (emotion scores as float lists)
def format_labels(example):
    example["labels"] = [example[emo] for emo in emotion_cols]
    return example

dataset = dataset.map(format_labels)

# Step 7: Set dataset format for PyTorch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 8: Train/test split (90% train, 10% eval)
split = dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# Step 9: Define training args
training_args = TrainingArguments(
    output_dir="./finetuned_emotion",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    # no evaluation_strategy or eval_steps
)



# Step 10: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Step 11: Train!
trainer.train()

# Step 12: Save your fine-tuned model + tokenizer
trainer.save_model("./finetuned_emotion")
tokenizer.save_pretrained("./finetuned_emotion")

print("✅ Fine-tuning complete! Model saved to ./finetuned_emotion")