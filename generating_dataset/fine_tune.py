import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy



# Load dataset
df = pd.read_csv('logs/2.csv')

# Define emotion columns (ensure these exist in the CSV)
emotion_cols = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Optional: Normalize scores to [0, 1] if not already
# df[emotion_cols] = df[emotion_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Load model and tokenizer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(emotion_cols)
    # No need to set problem_type manually
)

# Tokenization
def preprocess_function(examples):
    return tokenizer(examples["journal"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(preprocess_function, batched=True)

# Format labels
def format_labels(example):
    example["labels"] = [float(example[emo]) for emo in emotion_cols]
    return example

dataset = dataset.map(format_labels)

# Set dataset format
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Train-test split
split = dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_emotion",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",  # <--- Enables evaluation each epoch
)

# Define custom Trainer to use BCEWithLogitsLoss
from torch import nn

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


# Use the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./finetuned_emotion")
tokenizer.save_pretrained("./finetuned_emotion")

print("✅ Fine-tuning complete! Model saved to ./finetuned_emotion")
