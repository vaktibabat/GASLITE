# Fine-tune a model over a sample of MSMARCO BM25 and evaluate 3 metrics over the fine-tuning process:
# 1. Isotropy on the validation set
# 2. Robustness (PERFECT APPEARED@10) on multiple concepts
# 3. NanoBEIR NDCG@10
import torch
import argparse
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from src.benchmarking.robustness import RobustnessEvaluator
from src.benchmarking.performance import PerformanceEvaluator
from torch import optim
from sklearn.model_selection import train_test_split
from src.isotropy import MultiConceptIsotropyEvaluator
from IsoScore.IsoScore import *
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from src import data_utils
from tqdm import tqdm
# FOR REPRODUCABILITY
torch.random.manual_seed(100)
random.seed(100)
np.random.seed(100)
# ---SETUP---
concept_portion_to_train = 0.5
dataset_name = "msmarco"
data_split = "train-concepts"
data_portion = 1.0

device = torch.device("cuda")

# Possible models to evaluate
models = ["intfloat/e5-base-v2", "sentence-transformers/all-MiniLM-L6-v2", "Snowflake/snowflake-arctic-embed-m", "nomic-ai/nomic-embed-text-v1"]
# Concepts to evaluate robustness on
concepts = ["potter", "iphone", "vaccine"]
# Some models have specific query and passage prefixes
query_prefixes = {
    "intfloat/e5-base-v2": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "Snowflake/snowflake-arctic-embed-m": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1": "search_query: "
}
passage_prefixes = {
    "intfloat/e5-base-v2": "passage: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "Snowflake/snowflake-arctic-embed-m": "",
    "nomic-ai/nomic-embed-text-v1": "serach_passage: ",
}

model_name = None
tokenizer = None

def load_data(concept_to_attack=None, model_hf_name=None):
    if concept_to_attack is not None:
        with open(f"config/cover_alg/concept-{concept_to_attack}.yaml", "r") as f:
            import yaml
            concept_config = yaml.safe_load(f)
            concept_qids = concept_config['concept_qids']  # fetched from the attack config

        heldin_concept_qids, heldout_concept_qids = (concept_qids[:int(len(concept_qids)*concept_portion_to_train)],
                                                    concept_qids[int(len(concept_qids)*concept_portion_to_train):])

    # Load dataset:
    return data_utils.load_dataset(
        dataset_name=dataset_name,
        data_split=data_split,
        data_portion=data_portion,
        embedder_model_name=model_hf_name,
        filter_in_qids=None if concept_to_attack is None else concept_qids,
    )

# One fine-tuning batched GD step
def batch_step(
        model, # The model we want to fine-tune
        query_C0, pos_C0, neg_C0, # shrinkage matrices for the queries, hard positives, and hard negatives
        batch, # the batch of data
        reg, # I-STAR regularizer to use
        optimizer,
        criterion,
):
    h = model[0].auto_model.config.hidden_size
    for param in model.parameters():
        param.requires_grad = True

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
    pos_batch_iso = reg.IsoScore_star(pos_points, pos_C0, zeta=config.zeta, gpu_id=0, is_eval=False)
    neg_batch_iso = reg.IsoScore_star(neg_points, neg_C0, zeta=config.zeta, gpu_id=0, is_eval=False)
    # Avg. the isotropy scores and evaluate the complete loss
    avg_batch_iso = (query_batch_iso + pos_batch_iso + neg_batch_iso) / 3
    loss = l_ce + config.tuning_param * (1 - avg_batch_iso)

    loss.backward()
    optimizer.step()

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

def train_preprocess_function(examples):
    query_texts = [query_prefixes[model_name] + text for text in examples["query"]]
    pos_texts = [passage_prefixes[model_name] + text for text in examples["positive"]]
    neg_texts = [passage_prefixes[model_name] + text for text in examples["negative"]]
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
    texts = [query_prefixes[model_name] + text for text in examples["query"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

def positive_preprocess_function(examples):
    texts = [passage_prefixes[model_name] + text for text in examples["positive"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

def negative_preprocess_function(examples):
    texts = [passage_prefixes[model_name] + text for text in examples["negative"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

def prepare_dataset(config):
    dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet")
    train_dataset = dataset["train"]

    train_df = pd.DataFrame({"query": train_dataset["query"], "positive": train_dataset["positive"], "negative": train_dataset["negative"]})
    # Use a limited amount of samples
    train_df, _ = train_test_split(train_df, train_size=config.train_size, random_state=100)

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
    # Create a robustness evaluator for each concept
    concepts_to_eval = ["potter", "iphone", "vaccine"]
    robustness_evals = {k: RobustnessEvaluator(config.model_hf_name, k) for k in concepts_to_eval}
    perf_eval = PerformanceEvaluator(config.model_hf_name)
    iso_eval = MultiConceptIsotropyEvaluator(concepts_to_eval, config.model_hf_name)
    model.train()
    # Set up loss & optimizer
    criterion = MultipleNegativesRankingLoss(model)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    # Prepare datasets
    train_dataset, query_dataset, pos_dataset, neg_dataset = prepare_dataset(config)
    # Prepare training set dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # Prepare dataloaders for each column; these are used to evaluate the $\Sigma_{S_i}$ matrix at each epoch
    query_dataloader = DataLoader(query_dataset, batch_size=config.batch_size)
    pos_dataloader = DataLoader(pos_dataset, batch_size=config.batch_size)
    neg_dataloader = DataLoader(neg_dataset, batch_size=config.batch_size)
    # Start training
    reg = istar()
    model.train()
    # Metrics after each epoch
    iso_results_all = []
    robustness_results_all = []
    ndcg_results_all = []

    for epoch in range(config.epochs):
        # Compute shrinkage matrices
        query_C0 = compute_shrinkage_matrix(query_dataloader, model)
        pos_C0 = compute_shrinkage_matrix(pos_dataloader, model)
        neg_C0 = compute_shrinkage_matrix(neg_dataloader, model)

        loop = tqdm(train_dataloader, leave=True)
        loop.set_description(f"Epoch {epoch+1}")

        for batch in loop:
            batch_step(model, query_C0, pos_C0, neg_C0,
                       batch, reg, optimizer, criterion)

        # Evaluate metrics
        iso_results = iso_eval.evaluate(model)
        robustness_results = {concept: robustness_evals[concept].evaluate(model) for concept in concepts_to_eval}
        ndcg = perf_eval.evaluate(model)

        iso_results_all += [iso_results]
        robustness_results_all += [robustness_results]
        ndcg_results_all += [ndcg]

        print(iso_results)
        print(robustness_results)
        print(ndcg)

    print(iso_results_all)
    print(robustness_results_all)
    print(ndcg_results_all)


def main(config):
    global model_name, tokenizer

    # Pre-load MSMARCO so that we won't have to reload it in each evaluation
    model_name = config.model_hf_name
    model = SentenceTransformer(config.model_hf_name)
    tokenizer = model.tokenizer

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
    parser.add_argument("--concept", default="potter", type=str)
    parser.add_argument("--train_size", default=1000, type=int)

    config = parser.parse_args()

    main(config)