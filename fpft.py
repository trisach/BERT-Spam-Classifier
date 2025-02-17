import torch
import torch.nn
from utils import prepare_datasets
from transformers import DistilBertForSequenceClassification,AutoTokenizer
from transformers import TrainingArguments,Trainer
import json
from sklearn.metrics import f1_score
import os

os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = 2)
model.to(device)

params = sum(p.numel() for p in model.parameters())
print(f'Model Parameter count : {params}')
for p in model.parameters(): 
    p.requires_grad_(True)

train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters : {train_params}')

tokenized_dataset,test_dataloader = prepare_datasets("train1.csv","validation1.csv","test1.csv",tokenizer)

args = TrainingArguments(
    output_dir = 'fpft',
    num_train_epochs = 5,
    learning_rate = 2e-5,
    eval_strategy = "steps",
    eval_steps = 20,
    save_strategy = "epoch",
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32)
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["validation"],
    tokenizer = tokenizer
)
trainer.train()

def calc_batch_loss(model,input_ids,attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    return loss, logits

eval_loss = 0
correct = 0
total = 0
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, logits = calc_batch_loss(model,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        eval_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = (correct/total) * 100
metrics = trainer.state.log_history
training_losses = [entry['eval_loss'] for entry in metrics if 'eval_loss' in entry]
steps = [entry['step'] for entry in metrics if 'step' in entry]

f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"F1 Score: {f1:.4f}")

combined_metrics = {
    "training_losses": training_losses,
    "step": steps,
    "test_accuracy":test_acc,
    "num_params" : train_params,
    "f1score" : f1
}

with open("fpft_metrics.json", "w") as f:
    json.dump(combined_metrics, f, indent=4)
    print("Successfully saved metrics to training_metrics.json")
   