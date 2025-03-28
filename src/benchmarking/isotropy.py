import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
from IsoScore.IsoScore import *

device = torch.device("cuda")

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
        #batch = {key: value.to(device) for key, value in batch.items()}

        # Set model to eval and run input batches with no_grad to disable gradient calculations
        with torch.no_grad():
            outputs = model[0].auto_model(input_ids=batch[0], attention_mask=batch[1], output_hidden_states=True)
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

class IsotropyEvaluator:
    def __init__(self,
                 model,
                 qp_pairs_dataset,
                 corpus, queries, qrels,
                 results: dict):
        self.model = model
        self.tokenizer = model.tokenizer
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.results = results
        self.qp_pairs_dataset = qp_pairs_dataset
        self.sim_func = "cos_sim"

    def gaslite_cosreg(self, x='passage', y='passage', n_evals=1500, batch_size=100):
        """Calculate the average similarity (`sim_func`) of a random query to a random passage."""
        assert x in ['query', 'passage'] and y in ['query', 'passage']
        random.seed(101)
        x_texts = random.sample(self.qp_pairs_dataset[x].copy(), n_evals)
        y_texts = random.sample(self.qp_pairs_dataset[y].copy(), n_evals)
        #x_texts = self.qp_pairs_dataset[x].copy()[:n_evals]
        #y_texts = self.qp_pairs_dataset[y].copy()[-n_evals:]
        n_evals = min(n_evals, len(x_texts), len(y_texts))
        lst_sim_to_rand = []

        for _ in range(0, n_evals, batch_size):
            x_batch = self.model.encode(random.choices(x_texts, k=batch_size), convert_to_tensor=True)
            y_batch = self.model.encode(random.choices(y_texts, k=batch_size), convert_to_tensor=True)
            # if self.sim_func == 'cos_sim':  # then normalize before dot product  [WE CURRENTLY EXAMINE COS-SIM FOR ALL]
            x_batch = F.normalize(x_batch, p=2, dim=-1)
            y_batch = F.normalize(y_batch, p=2, dim=-1)
            curr_sim = torch.matmul(x_batch, y_batch.T)  # calculate the (pairwise) similarity matrix

            # Discard diagonal and flatten
            curr_sim = curr_sim[~torch.eye(curr_sim.shape[0]).bool()].flatten()
            lst_sim_to_rand.extend(curr_sim.tolist())

        return sum(lst_sim_to_rand) / len(lst_sim_to_rand)
    # Evaluate CosReg on a given dataset
    def cosreg_eval(self):   
        pass
    # Evaluate IsoScore* on a given dataset
    def iso_score_star(self, x="passage", n_evals=1500, batch_size=100, zeta=0.2):
        random.seed(101)
        samples = random.sample(self.qp_pairs_dataset[x].copy(), n_evals)
        texts = self.tokenizer(samples, return_tensors="pt", padding=True, truncation=True)
        texts = TensorDataset(texts["input_ids"].to(device), texts["attention_mask"].to(device))
        dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)
        h = self.model[0].auto_model.config.hidden_size 
        reg = istar()
        C0 = compute_shrinkage_matrix(dataloader, self.model)
        isos = []

        for idx, batch in enumerate(dataloader):
            outputs = self.model[0].auto_model(input_ids=batch[0], attention_mask=batch[1], output_hidden_states=True)
            points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,h))
            batch_iso = reg.IsoScore_star(points, C0, zeta=zeta, gpu_id=0, is_eval=False)
            isos += [batch_iso]
        
        # Return the mean IsoScore* across all batches
        return sum(isos) / len(isos)