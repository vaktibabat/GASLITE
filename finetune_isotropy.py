# Finetune an embedding model using the I-STAR regularizer to increase its representations' isotropy
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from IsoScore.IsoScore import *
import argparse
import os
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# possible models to finetune
model_hf_names = [
    "intfloat/e5-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/all-mpnet-base-v2",
    "Snowflake/snowflake-arctic-embed-m",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/msmarco-roberta-base-ance-firstp"
]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }

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
       # batch = {key: value.to(device) for key, value in batch.items()}

        # Set model to eval and run input batches with no_grad to disable gradient calculations
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
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

# Finetune an AutoModel and increase its Isotropy using I-STAR regularization
def fine_tune_model(config):
    model = AutoModel.from_pretrained(config.model_hf_name).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr)
    tokenizer = AutoTokenizer.from_pretrained(config.model_hf_name)
    reg = istar()
    # I-STAR config
    zeta = config.zeta
    tuning_param = config.tuning_param
    dataset = TextDataset(open(config.dataset_path).readlines(), tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size)
    h = model.config.hidden_size

    losses = []

    for epoch in range(config.epochs):
        C0 = compute_shrinkage_matrix(train_dataloader, model)

        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Get the hidden states
            outputs = model(**batch, output_hidden_states=True)
            points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,h))
            batch_iso  = reg.IsoScore_star(points, C0, zeta=zeta, gpu_id=0, is_eval=False)
            iso_score_loss = tuning_param * (1 - batch_iso)

            iso_score_loss.backward()
            optimizer.step()

        # Save the model in the directory specified in the config
        torch.save(model.state_dict(), os.path.join(config.output_path, f"{config.model_hf_name.split('/')[1]}-{epoch}.pt"))

        losses.append((epoch, iso_score_loss))

    return losses

def main(config):
    if config.model_hf_name not in model_hf_names:
        print(f"Model must be one of {model_hf_names}")
        return

    fine_tune_model(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--model_hf_name", default=None, type=str)
    parser.add_argument("--zeta", default=0.2, type=float)
    parser.add_argument("--tuning_param", default=0.25, type=float)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)

    config = parser.parse_args()

    main(config)