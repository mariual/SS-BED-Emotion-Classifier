import numpy as np
import torch

def load_sswe_embeddings(sswe_path):
    """
    Load SSWE embeddings from a .txt file with lines:
      token dim1 dim2 ... dim50
    Returns a dict: token -> 50D numpy array
    """
    sswe_dict = {}
    with open(sswe_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 51:
                continue
            word = vals[0]
            vec = np.array(vals[1:], dtype=float)
            sswe_dict[word] = vec
    return sswe_dict

def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings. Each line:
       token dim1 dim2 ... dimN
    (Here N=100 for glove.6B.100d.txt)
    """
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            word = vals[0]
            vec = np.array(vals[1:], dtype=float)
            glove_dict[word] = vec
    return glove_dict

def build_embedding_matrices(word2idx, sswe_emb, glove_emb, sswe_dim=50, glove_dim=100):
    """
    Build two embedding matrices (for SSWE and GloVe) aligned to the same word2idx.
    shape:
       - SSWE embedding: (vocab_size, sswe_dim)
       - GloVe embedding: (vocab_size, glove_dim)
    We will pass these to two separate embedding layers in the model.
    """
    vocab_size = len(word2idx)
    sswe_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, sswe_dim))
    glove_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, glove_dim))
    
    # For special tokens:
    # <pad> -> zero vector, <unk> -> random, or you can keep them random
    # Here we do 0 for pad, random for unk
    sswe_matrix[0] = np.zeros((sswe_dim,))   # <pad>
    glove_matrix[0] = np.zeros((glove_dim,)) # <pad>

    for token, idx in word2idx.items():
        if token in sswe_emb:
            sswe_matrix[idx] = sswe_emb[token]
        if token in glove_emb:
            glove_matrix[idx] = glove_emb[token]

    # Convert to PyTorch tensors
    sswe_tensor = torch.tensor(sswe_matrix, dtype=torch.float)
    glove_tensor = torch.tensor(glove_matrix, dtype=torch.float)
    return sswe_tensor, glove_tensor

