import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformer
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
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
# Load BM-25
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet")

def train_preprocess_function(examples):
    query_texts = ["query: " + text for text in examples["query"]]
    pos_texts = ["passage: " + text for text in examples["positive"]]
    neg_texts = ["passage: " + text for text in examples["negative"]]
    enc_queries = tokenizer(query_texts, truncation=True, padding="max_length", max_length=256)
    enc_pos = tokenizer(pos_texts, truncation=True, padding="max_length", max_length=256)
    enc_neg = tokenizer(neg_texts, truncation=True, padding="max_length", max_length=256)

    return {
        "query_input_ids": enc_queries["input_ids"],
        "query_attn_mask": enc_queries["attention_mask"],
        "pos_input_ids": enc_pos["input_ids"],
        "pos_attn_mask": enc_pos["attention_mask"],
        "neg_input_ids": enc_neg["input_ids"],
        "neg_attn_mask": enc_neg["attention_mask"],
    }

# Preprocess the queries
def query_preprocess_function(examples):
    texts = ["query: " + text for text in examples["query"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

def positive_preprocess_function(examples):
    texts = ["passage: " + text for text in examples["positive"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

def negative_preprocess_function(examples):
    texts = ["passage: " + text for text in examples["negative"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

# Compute the shrinkage matrix $\Sigma_{S_i}$ at epoch i
# slightly modified version of get_ci from https://github.com/bcbi-edu/p_eickhoff_isoscore/blob/main/I-STAR/training_utils.py#L40
def compute_shrinkage_matrix(data, model, max_points=250000):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix. Run this as a warmup to generate a stable
    covariance matrix for IsoScore Regularization"""
    num_points = 0
    points_list = []
    model.eval()
    h = model[0].auto_model.config.hidden_size

    for idx, batch in enumerate(data):
        # send batch to device
        batch = {key: value.to(device) for key, value in batch.items()}

        # Set model to eval and run input batches with no_grad to disable gradient calculations
        with torch.no_grad():

            outputs = model[0].auto_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
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

    train_df = pd.DataFrame({"query": train_dataset["query"], "positive": train_dataset["positive"], "negative": train_dataset["negative"]})

    # Use only 1000 samples
    train_df, _ = train_test_split(train_df, train_size=1000, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(train_preprocess_function, batched=True)
    query_dataset = train_dataset.map(query_preprocess_function, batched=True)
    pos_dataset = train_dataset.map(positive_preprocess_function, batched=True)
    neg_dataset = train_dataset.map(negative_preprocess_function, batched=True)

    train_dataset.set_format(type="torch", columns=["query_input_ids", "query_attn_mask",
                                                    "pos_input_ids", "pos_attn_mask",
                                                    "neg_input_ids", "neg_attn_mask"])
    query_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    pos_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    neg_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return train_dataset, query_dataset, pos_dataset, neg_dataset

def fine_tune_model(model, config):
    # Set up loss & optimizer
    criterion = MultipleNegativesRankingLoss(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    # Prepare datasets
    train_dataset, query_dataset, pos_dataset, neg_dataset = prepare_dataset()
    # Prepare training set dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # Prepare dataloaders for each column; these are used to evaluate the $\Sigma_{S_i}$ matrix at each epoch
    query_dataloader = DataLoader(query_dataset, batch_size=16)
    pos_dataloader = DataLoader(pos_dataset, batch_size=16)
    neg_dataloader = DataLoader(neg_dataset, batch_size=16)
    # Start training
    model_name = config.model_hf_name.split("/")[0]
    reg = istar()
    model.train()
    h = model[0].auto_model.config.hidden_size 
    
    for epoch in range(config.epochs):
        query_C0 = compute_shrinkage_matrix(query_dataloader, model)
        pos_C0 = compute_shrinkage_matrix(pos_dataloader, model)
        neg_C0 = compute_shrinkage_matrix(neg_dataloader, model)

        print(f"Epoch {epoch + 1} / {config.epochs}")
        train_loss = 0

        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU
            optimizer.zero_grad()
            # In addition to evaluating the retrieval loss, we need to run the model on each individual query/passage
            # to get the hidden states, allowing us to evaluate the batch isotropy
            query_outputs = model[0].auto_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attn_mask"], output_hidden_states=True)
            pos_outputs = model[0].auto_model(input_ids=batch["pos_input_ids"], attention_mask=batch["pos_attn_mask"], output_hidden_states=True)
            neg_outputs = model[0].auto_model(input_ids=batch["neg_input_ids"], attention_mask=batch["neg_attn_mask"], output_hidden_states=True)
            # Evaluate retrieval loss
            l_ce = criterion(sentence_features=[{"input_ids": batch["query_input_ids"], "attention_mask": batch["query_attn_mask"]},
                                                {"input_ids": batch["pos_input_ids"], "attention_mask": batch["pos_attn_mask"]},
                                                {"input_ids": batch["neg_input_ids"], "attention_mask": batch["neg_attn_mask"]}],
                                                labels=None)
            # Concatenate hidden states into points
            query_points = torch.reshape(torch.stack(query_outputs.hidden_states)[1:,:,:,:], (-1,h))
            pos_points = torch.reshape(torch.stack(pos_outputs.hidden_states)[1:,:,:,:], (-1,h))
            neg_points = torch.reshape(torch.stack(neg_outputs.hidden_states)[1:,:,:,:], (-1,h))
            # Evaluate batch isotropy on the query and the two passages
            query_batch_iso = reg.IsoScore_star(query_points, query_C0, zeta=config.zeta, gpu_id=0, is_eval=False)
            pos_batch_iso = reg.IsoScore_star(pos_points, query_C0, zeta=config.zeta, gpu_id=0, is_eval=False)
            neg_batch_iso = reg.IsoScore_star(neg_points, query_C0, zeta=config.zeta, gpu_id=0, is_eval=False)
            # Avg. the isotropy scores and evaluate the complete loss
            avg_batch_iso = (query_batch_iso + pos_batch_iso + neg_batch_iso) / 3
            loss = l_ce + config.tuning_param * (1 - avg_batch_iso)

            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), os.path.join(f"models_with_benign", f"{model_name}_msmarco_istar_{config.tuning_param}_{epoch}.pt"))

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")

def main(config):
    global tokenizer

    if config.model_hf_name not in model_hf_names:
        print(f"Model must be one of {model_hf_names}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_hf_name)
    model = SentenceTransformer(config.model_hf_name)

    fine_tune_model(model, config)

# Train and evaluate
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
