import torch
import torch.nn as nn
import pandas as pd
import math
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import DistilBertModel,DistilBertTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import prepare_datasets

num_epochs = 5
learning_rate = 2e-6
alpha = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class lora(nn.Module):
    def __init__(self,olayer,rank=8):
        super().__init__()
        self.rank = rank
        self.original_layer = olayer
        self.alpha = alpha
        self.in_features,self.out_features = self.original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.empty(self.in_features,self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank,self.out_features))
        model.out_head = nn.Linear(model.config.hidden_size , 2)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        
    def forward(self,hidden_states):
        lora_w_out = hidden_states @ self.lora_A @ self.lora_B
        return self.original_layer(hidden_states) + (self.alpha) * lora_w_out
    
def add_lora(model,rank=8):
    for i in range(model.config.num_hidden_layers):
        attn = model.transformer.layer[i].attention
        attn.q_lin = lora(attn.q_lin,rank)
        attn.v_lin = lora(attn.v_lin,rank)
        attn.k_lin = lora(attn.k_lin,rank)
        attn.out_lin = lora(attn.out_lin,rank)
    return model

def calc_batch_loss(model,input_ids,attention_mask, labels):
    logits = model(input_ids = input_ids,attention_mask = attention_mask)
    logits_cls = logits.last_hidden_state[:,0,:]
    classifier = model.out_head(logits_cls)
    loss = nn.functional.cross_entropy(classifier,labels)
    return loss,classifier

tokenized_dataset, test_dataloader = prepare_datasets("./data/train1.csv","./data/validation1.csv","./data/test1.csv",tokenizer=tokenizer)

#training loop
if __name__ == "__main__":
    lora_model = add_lora(model,64)
    lora_model.to(device)
    train_params1 = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f'Before freezing number of trainable parameters(LoRA) : {train_params1}')
    
    for name,p in lora_model.named_parameters():
        if 'lora' not in name:
            p.requires_grad = (False)
    lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f'Number of LoRA parameters : {lora_params}')

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True,collate_fn=collator)
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=32,collate_fn=collator)


    optimizer = AdamW(lora_model.parameters(), lr=learning_rate,weight_decay = 0.01)

    total_steps = len(train_dataloader) * 5
    print(total_steps) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # Training loop
    train_loss = []
    step = []
    import time
    stime = time.time()
    step_count = 0 
    for epoch in range(num_epochs):  
        lora_model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, logits = calc_batch_loss(lora_model,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            if step_count % 10 == 0 or step_count==total_steps:
                print(f'Recording : {step_count}')
                train_loss.append(loss.item())
                step.append(step_count)
            step_count += 1

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        with torch.no_grad():
            lora_model.eval()
            eval_loss = 0
            correct = 0
            total = 0
                
            for batch in eval_dataloader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
            
                        loss, logits = calc_batch_loss(lora_model,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        eval_loss += loss.item()
                        predictions = torch.argmax(logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
        print(f'Val accuracy : {(correct/total)*100 :.4f} | Val Loss : {eval_loss/len(eval_dataloader):.4f}')
    
        lora_model.train()
        checkpoint_path = f"distilbert_lora_epoch_{epoch+1}.pth"
        torch.save(lora_model, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    etime = time.time()
    print(f'Total time taken to train : {(etime - stime)}')
    for batch in test_dataloader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
            
                        loss, logits = calc_batch_loss(lora_model,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        eval_loss += loss.item()
                        predictions = torch.argmax(logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
    test_acc = (correct/total) * 100
    all_preds = []
    all_labels = []
    from sklearn.metrics import f1_score
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss,logits = calc_batch_loss(model,input_ids,attention_mask,labels) 
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
f1 = f1_score(all_labels,all_preds,average = 'weighted')
print(f'F1 score : {f1}')
import json
combined_metrics = {
    "train_loss" : train_loss,
    "step" : step,
    "test_accuracy" : test_acc,
    "num_params" : lora_params,
    "f1score" : f1   
}
with open("lora_metrics.json","w") as f:
      json.dump(combined_metrics,f,indent = 4)
