import torch
from sklearn.metrics import f1_score
from transformers import DistilBertForSequenceClassification,AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import get_peft_model
from transformers import TrainingArguments,Trainer
from peft import LoraConfig
from utils import prepare_datasets
import json
import os
num_epochs = 5
alpha = 16
lr = 2e-4
os.environ["WANDB_DISABLED"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calc_batch_loss(model,input_ids,attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    return loss, logits

config = BitsAndBytesConfig(load_in_4bit = True,
                            bnb_4bit_quant_type = "nf4",
                            bnb_4bit_use_double_quant = True,
                            bnb_4bit_compute_dtype = torch.float16)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = 2)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model.to(device)
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_lin","k_lin","v_lin","out_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS")
model = get_peft_model(model, config)

tokenized_dataset, test_dataloader = prepare_datasets(
    "./data/train1.csv",
    "./data/validation1.csv",
    "./data/test1.csv",
    tokenizer
)
num_params = sum(p.numel() for name,p in model.named_parameters() if "lora" in name)
print("Number of trainable parameters : ",num_params)
training_args = TrainingArguments(
    save_strategy = "epoch",
    learning_rate = lr,
    output_dir="out", 
    evaluation_strategy="steps", 
    eval_steps=10, 
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,   
    num_train_epochs=num_epochs,  
    lr_scheduler_type = "cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer)

trainer.train()
eval_loss = 0
correct = 0
total = 0
all_preds = []
all_labels = []
for batch in test_dataloader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    loss, logits = calc_batch_loss(model,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    eval_loss += loss.item()
    predictions = torch.argmax(logits, dim=-1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)
    preds = torch.argmax(logits, dim=-1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

test_accuracy = (correct / total) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"F1 Score: {f1:.4f}")

metrics = trainer.state.log_history
training_losses = [element['eval_loss'] for element in metrics if 'eval_loss' in element]
steps = [element["step"] for element in metrics if 'step' in element]
print(training_losses)
combined_metrics = {
    "training_losses": training_losses,
    "steps" : steps,
    "test_accuracy": test_accuracy,
    "num_params" : num_params,
    "f1score" : f1
}

with open("qlora_metrics.json", "w") as f:
    json.dump(combined_metrics, f, indent=4)
    print("Successfully saved metrics to qlora_metrics.json")
