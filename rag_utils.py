import torch
import numpy as np
from tqdm import tqdm
import os
from abc import ABC, abstractmethod
import faiss

class EmbeddingModel(ABC):
    @abstractmethod
    def get_emb_dim(self) -> int:
        """Returns a molecular embedding size."""
        pass
    
    @abstractmethod
    def embed(self, smiles: list) -> np.ndarray:
        """Embeds a list of SMILES strings into a numpy array."""
        pass
    
from transformers import AutoTokenizer, AutoModel

class ChemBERTa(EmbeddingModel):
    def __init__(self, chemberta_path, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(chemberta_path)
        self.model = AutoModel.from_pretrained(chemberta_path).to(self.device)

        self.emb_dim = self.model.config.hidden_size
        self.batch_size = batch_size

    def get_emb_dim(self):
        return self.emb_dim

    def embed(self, smiles):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(smiles), self.batch_size):
                batch_smiles = smiles[i:i+self.batch_size]
                inputs = self.tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        if not embeddings:
            return np.array([])    
        return np.concatenate(embeddings, axis=0)

def _create_emb_model(name, params=None):
    params = params or {}
    return ChemBERTa(**params)
    
class MolRAG:
    def __init__(self, args):
        emb_model_params = {}
        emb_model_params['chemberta_path'] = args.chemberta_path
        if hasattr(args, 'batch_size_chemberta'):
            emb_model_params['batch_size'] = args.batch_size_chemberta
        elif hasattr(args, 'batch_size'):
            emb_model_params['batch_size'] = args.batch_size
        self.emb_model = _create_emb_model(emb_model_params)
        self.smi_list = None
        self.property = None
        self.prop_names = None # 'homo', 'lumo', 'alpha', 'gap', 'mu', 'Cv'
        self.mean = None
        self.std = None
        self.faiss_index = None


    def build_db(self, data, prop_keys):
        self.smi_list = data['smiles']
        self.prop_names = prop_keys

        prop_list = []
        for key in self.prop_names:
            val = data[key]
            if isinstance(val, torch.Tensor):
                prop_list.append(val.view(-1, 1))
            elif isinstance(val, np.ndarray):
                prop_list.append(torch.from_numpy(val).float().view(-1, 1))
            else: # Assuming it's a list or scalar, convert to tensor
                prop_list.append(torch.tensor(val).float().view(-1, 1))

        stacked_prop = torch.cat(prop_list, dim=1)
        self.props = stacked_prop.numpy().astype(np.float32)

        self.mean = np.mean(self.props, axis=0)
        self.std = np.std(self.props, axis=0)
        self.std[self.std == 0] = 1.0
        self.prop_norm = (self.props - self.mean) / self.std

        D = self.prop_norm.shape[1]
        self.faiss_idx = faiss.IndexFlatL2(D)
        self.faiss_idx.add(self.prop_norm)

    def _norm_target(self, targets: np.ndarray) -> np.ndarray:
        """Normalizes a batch of target property values."""
        return (targets - self.mean) / self.std

    def retrieve(self, targets: dict, k: int = 10):
        target_vals = np.array([targets[prop] for prop in self.prop_names], dtype=np.float32).reshape(1, -1)
        ret_mol_embs_batch, ret_prop_vals_batch, ret_dist_to_targs_batch, ret_smi_batch = self.retrieve_batch(target_vals, k)
        return ret_mol_embs_batch[0], ret_prop_vals_batch[0], ret_dist_to_targs_batch[0], ret_smi_batch[0]


    def retrieve_batch(self, batch_targs: np.ndarray, k: int = 10):
        batch_size = batch_targs.shape[0]
        norm_batch_targs = self._norm_target(batch_targs) # (bs, num_props)

        if self.faiss_idx is not None:
            n_ret = k + 1
            dists_sq, top_k_idx = self.faiss_idx.search(norm_batch_targs, n_ret)

            dists_sq = dists_sq[:, 1:]
            top_k_idx = top_k_idx[:, 1:]
        else:
            # Fallback for when faiss_idx is None, though it should be built
            top_k_idx = np.zeros((batch_size, k), dtype=int)
            dists_sq = np.zeros((batch_size, k), dtype=np.float32)
            for i in range(batch_size):
                all_dists_sq = np.sum((self.prop_norm - norm_batch_targs[i])**2, axis=1)
                partitioned_indices = np.argpartition(all_dists_sq, k)
                top_k_idx[i] = partitioned_indices[:k]
                dists_sq[i] = all_dists_sq[top_k_idx[i]]
            
        flat_top_k_idx = top_k_idx.flatten()
        unique_indices, inv_indices = np.unique(flat_top_k_idx, return_inverse=True)
        unique_smiles = [self.smi_list[i] for i in unique_indices]

        if not unique_smiles:
            empty_emb_dim = self.emb_model.get_emb_dim()
            ret_mol_embs = torch.zeros((batch_size, k, empty_emb_dim), dtype=torch.float32)
            ret_prop_vals = torch.zeros((batch_size, k, len(self.prop_names)), dtype=torch.float32)
            ret_dist_to_targs = torch.zeros((batch_size, k, len(self.prop_names)), dtype=torch.float32)
            retrieved_smiles_list = [[] for _ in range(batch_size)]
            return ret_mol_embs, ret_prop_vals, ret_dist_to_targs, retrieved_smiles_list

        unique_embs = self.emb_model.embed(unique_smiles)
        unique_embs_t = torch.from_numpy(unique_embs).float()

        ret_mol_embs = torch.zeros((batch_size, k, self.emb_model.get_emb_dim()), dtype=torch.float32)
        ret_prop_vals = torch.zeros((batch_size, k, len(self.prop_names)), dtype=torch.float32)
        ret_dist_to_targs = torch.zeros((batch_size, k, len(self.prop_names)), dtype=torch.float32)
        retrieved_smiles_list = []

        for i in range(batch_size):
            smi_for_mol = []
            for j in range(k):
                idx = top_k_idx[i, j]
                idx_in_unique = inv_indices[i * k + j] 

                ret_mol_embs[i, j] = unique_embs_t[idx_in_unique]
                ret_prop_vals[i, j] = torch.from_numpy(self.props[idx]).float()

                # Calculate distance to target for each property (using current target value for this batch item)
                abs_diff = torch.abs(ret_prop_vals[i, j] - torch.from_numpy(batch_targs[i]).float())
                ret_dist_to_targs[i, j] = abs_diff

                smi_for_mol.append(self.smi_list[idx])
            retrieved_smiles_list.append(smi_for_mol)
        # (bs, k, emb_dim), (bs, k, prop), (bs, k, prop)
        return ret_mol_embs, ret_prop_vals, ret_dist_to_targs, retrieved_smiles_list
    

    
