import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse
import scipy.io
from sentence_transformers import SentenceTransformer
from . import file_utils
import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def csr_rows_to_coo_tensor(csr, r0, r1, device):
    """Slice rows [r0:r1) from SciPy CSR -> PyTorch sparse COO (r1-r0, V)."""
    sub = csr[r0:r1].tocoo()
    if sub.nnz == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2,0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            (r1 - r0, csr.shape[1]),
            device=device
        ).coalesce()
    idx = torch.stack([
        torch.from_numpy(sub.row.astype(np.int64)),
        torch.from_numpy(sub.col.astype(np.int64))
    ], dim=0).to(device)
    val = torch.from_numpy(sub.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(idx, val, (r1 - r0, csr.shape[1]), device=device).coalesce()


class JointDocSubdocDataset(Dataset):
    """
    For doc i, returns:
      - 'data': dense doc BoW (V,)
      - 'sub_bow': sparse COO (S, V) with all subdocs of that doc
    """
    def __init__(self, doc_bow_dense_torch, sub_csr, S, device='cpu'):
        assert isinstance(doc_bow_dense_torch, torch.Tensor)
        self.doc = doc_bow_dense_torch           # (N, V) torch tensor (on any device)
        self.N = self.doc.shape[0]
        self.S = S
        self.sub_csr = sub_csr                   # (N*S, V) SciPy CSR
        self.sparse_device = torch.device(device)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        r0, r1 = idx * self.S, (idx + 1) * self.S
        sub = csr_rows_to_coo_tensor(self.sub_csr, r0, r1, self.sparse_device)  # (S, V) sparse
        return {'data': self.doc[idx], 'sub_bow': sub}

class JointDocSubdocIndexDataset(Dataset):
    """
    Returns only:
      - doc_bow row (numpy float32, stays CPU)
      - doc_idx (int)
    The collate_fn will build the subdoc sparse batch using CSR arrays.
    """
    def __init__(self, doc_bow_np, N, S):
        self.doc_bow_np = doc_bow_np  # (N, V) numpy float32
        self.N = N
        self.S = S

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {'data': self.doc_bow_np[idx], 'doc_idx': idx}

def collate_joint_fast_cpu(batch, csr, S):
    """
    CPU-only collate that COMPRESSES the (B*S, V) subdoc matrix to (M, V)
    keeping only rows with nnz > 0.
    Returns:
      - data: (B, V) CPU float32
      - sub_rows/sub_cols/sub_vals: COO arrays for (M, V)
      - sub_shape: (M, V)
      - row2doc: length-M vector mapping subdoc-row -> doc index in [0..B-1]
      - B
    """
    B = len(batch)
    V = csr.shape[1]

    # Dense doc BoW on CPU
    data_np = np.stack([b['data'] for b in batch], axis=0).astype(np.float32, copy=True)
    data = torch.from_numpy(data_np)  # (B, V) CPU

    indptr  = csr.indptr
    indices = csr.indices
    values  = csr.data

    out_rows_list, out_cols_list, out_vals_list = [], [], []
    row2doc_list = []
    next_row = 0

    doc_indices = [b['doc_idx'] for b in batch]
    for i, d in enumerate(doc_indices):
        r0 = d * S
        r1 = r0 + S
        # counts per subdoc row
        counts = np.diff(indptr[r0:r1 + 1])  # shape (S,)

        # valid subdocs are those with nnz > 0
        valid_mask = counts > 0
        if not np.any(valid_mask):
            continue

        # For this doc, for each valid subdoc j, append its row
        # Compute the contiguous slice in CSR
        start, end = indptr[r0], indptr[r1]
        cols_this_doc = indices[start:end].astype(np.int64, copy=False)
        vals_this_doc = values[start:end].astype(np.float32, copy=False)

        # Expand per-subdoc rows
        # Build an array that assigns each nonzero entry to its subdoc row index
        # within [0..S-1], then filter by valid_mask and remap to compact rows
        per_row_counts = counts.astype(np.int64, copy=False)
        subdoc_ids_full = np.repeat(np.arange(S, dtype=np.int64), per_row_counts)

        # Keep only nonzero entries from valid subdocs
        keep = valid_mask[subdoc_ids_full]               # boolean mask over nnz
        cols_kept = cols_this_doc[keep]
        vals_kept = vals_this_doc[keep]
        subdoc_ids_kept = subdoc_ids_full[keep]

        # Map each kept subdoc j (0..S-1) to a compact row id
        # order is stable: iterate j where valid_mask[j] True
        remap = {j: (next_row + k) for k, j in enumerate(np.nonzero(valid_mask)[0])}
        row_ids_kept = np.vectorize(remap.get, otypes=[np.int64])(subdoc_ids_kept)

        out_rows_list.append(row_ids_kept)
        out_cols_list.append(cols_kept)
        out_vals_list.append(vals_kept)

        # Append row2doc for each valid subdoc row we created
        m_i = int(valid_mask.sum())
        row2doc_list.extend([i] * m_i)

        next_row += m_i

    if out_rows_list:
        sub_rows = torch.from_numpy(np.concatenate(out_rows_list, axis=0))
        sub_cols = torch.from_numpy(np.concatenate(out_cols_list, axis=0))
        sub_vals = torch.from_numpy(np.concatenate(out_vals_list, axis=0))
    else:
        sub_rows = torch.from_numpy(np.empty((0,), dtype=np.int64))
        sub_cols = torch.from_numpy(np.empty((0,), dtype=np.int64))
        sub_vals = torch.from_numpy(np.empty((0,), dtype=np.float32))

    row2doc = torch.from_numpy(np.asarray(row2doc_list, dtype=np.int64))  # (M,)

    return {
        'data': data,                    # (B, V) CPU
        'sub_rows': sub_rows,            # (nnz,) CPU int64
        'sub_cols': sub_cols,            # (nnz,) CPU int64
        'sub_vals': sub_vals,            # (nnz,) CPU float32
        'sub_shape': (int(row2doc.numel()), V),   # (M, V)
        'row2doc': row2doc,              # (M,)
        'B': B
    }

def load_contextual_embed(texts, device, model_name="all-mpnet-base-v2", show_progress_bar=True):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return embeddings

class BasicDatasetHandler:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False, device='cuda', as_tensor=False, contextual_embed=False, plm_model="all-mpnet-base-v2"):
        # train_bow: NxV
        # test_bow: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)
        self.plm_model = plm_model

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(
            self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if contextual_embed:
            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'train_bert.npz')):
                self.train_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'train_bert.npz'))['arr_0']
            else:
                self.train_contextual_embed = load_contextual_embed(
                    self.train_texts, device, model_name=self.plm_model)

            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'test_bert.npz')):
                self.test_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'test_bert.npz'))['arr_0']
            else:
                self.test_contextual_embed = load_contextual_embed(
                    self.test_texts, device, model_name=self.plm_model)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            
            self.train_data = self.train_bow
            self.test_data = self.test_bow

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)
            
            device_t = torch.device(device)

            # Doc data: already torch tensors because you did as_tensor=True earlier.
            # Move them back to CPU if needed to leverage pin_memory efficiently:
            self.train_data = self.train_data.cpu()
            self.test_data  = self.test_data.cpu()

            self.joint_train_dataset = JointDocSubdocIndexDataset(
                doc_bow_np=self.train_bow, N=self.N_tr, S=self.S_tr
            )
            self.joint_test_dataset = JointDocSubdocIndexDataset(
                doc_bow_np=self.test_bow,  N=self.N_te, S=self.S_te
            )

            def _collate_train(b): return collate_joint_fast_cpu(b, self.train_sub_csr, self.S_tr)
            def _collate_test(b):  return collate_joint_fast_cpu(b, self.test_sub_csr,  self.S_te)

            self.train_dataloader = DataLoader(
                self.joint_train_dataset,
                batch_size=batch_size,
                # shuffle=True,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                persistent_workers=True,
                pin_memory=True,          # let DataLoader do the pinning
                collate_fn=_collate_train,
            )
            self.test_dataloader = DataLoader(
                self.joint_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                persistent_workers=True,
                pin_memory=True,
                collate_fn=_collate_test,
            )

    def load_data(self, path, read_labels):

        self.train_bow = scipy.sparse.load_npz(
            f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(
            f'{path}/test_bow.npz').toarray().astype('float32')
        
        # ---- SUBDOCS: keep sparse CSR (N*S, V) ----
        self.train_sub_csr = scipy.sparse.load_npz(f"{path}/subspan/train_bow.npz")
        meta_tr = np.load(f"{path}/subspan/train_bow_meta.npz")
        self.N_tr, self.S_tr, self.V = int(meta_tr["N"]), int(meta_tr["S"]), int(meta_tr["V"])

        self.test_sub_csr = scipy.sparse.load_npz(f"{path}/subspan/test_bow.npz")
        meta_te = np.load(f"{path}/subspan/test_bow_meta.npz")
        self.N_te, self.S_te, V_te   = int(meta_te["N"]), int(meta_te["S"]), int(meta_te["V"])
        assert self.V == V_te, "Vocab size mismatch between train/test sub_bow"
        
        self.pretrained_WE = scipy.sparse.load_npz(
            f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        if read_labels:
            self.train_labels = np.loadtxt(
                f'{path}/train_labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')
