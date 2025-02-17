import pandas as pd
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset

def balance_dataset(df):
    subset = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n = subset, random_state = 123)
    bdf = pd.concat([ham_subset,df[df["Label"] == "spam"]])
    return bdf

def splitset(df,train,val):
    bdf = df.sample(frac = 1 , random_state = 123).reset_index(drop = True)
    print(bdf)
    train_split = int(train * bdf.shape[0])
    val_split = train_split +  int(val * bdf.shape[0])
    print(train_split,val_split)
    train_df = bdf[:train_split]
    val_df = bdf[train_split : val_split]
    test_df = bdf[val_split:]
    return train_df,val_df,test_df


def prepare_datasets(train_file, val_file, test_file, tokenizer):
    dataset = load_dataset("csv", data_files={
        "train": train_file,
        "validation": val_file,
        "test" : test_file
    })
    def tokenize(example):
        return tokenizer(example["Text"], truncation=True)
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.map(
        lambda example: {"labels": example["Label"], **example}
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["Label", "Text"])
    collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=10, collate_fn=collator)

    return tokenized_dataset, test_dataloader

if __name__ == "__main__":
    df = pd.read_csv("SMSSpamCollection.tsv",delimiter = "\t",names = ["Label","Text"])
    print(df)
    print(print(df["Label"].value_counts()))
    print("\nFixing imbalance in dataset\n")
    bdf = balance_dataset(df)
    bdf["Label"] = bdf["Label"].map({"spam" : 1,"ham" : 0,})
    train_df,val_df,test_df = splitset(bdf,0.89,0.05)
    train_df.to_csv("train1.csv", index=None)
    val_df.to_csv("validation1.csv", index=None)
    test_df.to_csv("test1.csv", index=None)
