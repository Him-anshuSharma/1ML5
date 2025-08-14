import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments
)
import torch.nn as nn

# ----------------- Config -----------------
MODEL_PATH = "./finetuned_emotion"
OUTPUT_DIR = "./finetuned_emotion_subtle"
CSV_FILE = "subtle_emotion_journals.csv"

LABEL_COLS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
# -----------------------------------------

# Load dataset
df = pd.read_csv(CSV_FILE)
dataset = Dataset.from_pandas(df)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Tokenization
def tokenize(batch):
    return tokenizer(batch["journal"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize, batched=True)

# Format labels
def format_labels(batch):
    batch["labels"] = torch.tensor([batch[c] for c in LABEL_COLS], dtype=torch.float)
    return batch

dataset = dataset.map(format_labels)
dataset = dataset.remove_columns(LABEL_COLS + ["journal"])

# Load model with regression config
config = AutoConfig.from_pretrained(MODEL_PATH)
config.problem_type = "multi_label_regression"
config.num_labels = len(LABEL_COLS)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,
)

# Custom Trainer with MSE loss
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete! Model saved to", OUTPUT_DIR)
