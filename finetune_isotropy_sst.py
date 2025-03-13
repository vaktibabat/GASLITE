import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
import os
import numpy as np
from IsoScore.IsoScore import *
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# possible models to fine-tune
model_hf_names = [
    "intfloat/e5-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/all-mpnet-base-v2",
    "Snowflake/snowflake-arctic-embed-m",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/msmarco-roberta-base-ance-firstp"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None

# Load SST-2 
dataset = load_dataset("glue", "sst2")

def preprocess_function(examples):
    texts = ["query: " + text for text in examples["sentence"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

# Compute the shrinkage matrix $\Sigma_{S_i}$ at epoch i
# slightly modified version of get_ci from https://github.com/bcbi-edu/p_eickhoff_isoscore/blob/main/I-STAR/training_utils.py#L40
def compute_shrinkage_matrix(data, model, max_points=250000):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix. Run this as a warmup to generate a stable
    covariance matrix for IsoScore Regularization"""
    num_points = 0
    points_list = []
    model.eval()
    h = model.config.hidden_size
    # main EVAL loop
    for idx, batch in enumerate(data):
        # send batch to device
        batch = {key: value.to(device) for key, value in batch.items()}

        # Set model to eval and run input batches with no_grad to disable gradient calculations
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
            points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,h))

        num_points += points.shape[0]

       # Collect the last state representations to a list and keep track of the number of points
        points = points.detach().cpu().numpy()
        points_list.append(points)

        if num_points > max_points:
            break
    # Convert model back to train mode:
    model.train()
    # Stack the points and calclate the sample covariance C0
    sample = np.vstack(points_list)
    C0 = np.cov(sample.T)
    return torch.tensor(C0, device=device)

def prepare_dataset():
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_df = pd.DataFrame({"sentence": train_dataset["sentence"], "label": train_dataset["label"]})

    # Use only 1000 random samples
    train_df, _ = train_test_split(train_df, train_size=1000, stratify=train_df["label"], random_state=42)

    # Tokenize queries
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, val_dataset

def fine_tune_model(model, config):
    train_dataset, val_dataset = prepare_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    # model name without the repo
    model_name = config.model_hf_name.split("/")[0]

    h = model.config.hidden_size 
    reg = istar()
    model.train()
    
    for epoch in range(config.epochs):
        C0 = compute_shrinkage_matrix(train_dataloader, model)
        print(f"Epoch {epoch + 1} / {config.epochs}")
        train_loss = 0

        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU

            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
            l_ce = criterion(outputs.logits, batch["label"])
            points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,h))
            batch_iso  = reg.IsoScore_star(points, C0, zeta=config.zeta, gpu_id=0, is_eval=False)
            loss = l_ce + config.tuning_param * (1 - batch_iso)

            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), os.path.join(config.output_path, f"{model_name}_sst-2_istar_{config.tuning_param}_{epoch}.pt"))

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")

def main(config):
    global tokenizer

    if config.model_hf_name not in model_hf_names:
        print(f"Model must be one of {model_hf_names}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_hf_name)

    model = AutoModelForSequenceClassification.from_pretrained(config.model_hf_name, num_labels=2).to(device)

    fine_tune_model(model, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--model_hf_name", default=None, type=str)
    parser.add_argument("--zeta", default=0.2, type=float)
    parser.add_argument("--tuning_param", default=0.25, type=float)
    parser.add_argument("--output_path", default=None, type=str)

    config = parser.parse_args()

    main(config)