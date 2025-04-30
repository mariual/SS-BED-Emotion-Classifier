import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re

# Separate BiLSTM for context and target utterance
class EmoContextDataset(Dataset):
    """
    A PyTorch Dataset for context and target utterances:
    This dataset uses 'turn1' and 'turn2' as the context, and 'turn3' as the target.
    Each sample includes:
    - context (concatenated turn1 + turn2)
    - target (turn3)
    - label (numeric: 0=others, 1=happy, 2=sad, 3=angry)
    
    The context and target sequences are padded or truncated to a fixed length.
    """
    def __init__(self, turn1s, turn2s, turn3s, labels, word2idx, max_len=30):
        self.turn1s = turn1s
        self.turn2s = turn2s
        self.turn3s = turn3s
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.turn3s)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index. The sample includes:
          - context (turn1 + turn2 concatenated and tokenized)
          - target (turn3 tokenized)
          - label (numeric)
        
        :param idx: Index of the sample to retrieve
        :return: Tuple containing context IDs, target IDs, and label
        """
        ctx_text = clean_text((self.turn1s[idx] or "") + " " + (self.turn2s[idx] or ""))
        tgt_text = clean_text(self.turn3s[idx])

        ctx_ids = [self.word2idx.get(t.lower(), self.word2idx["<unk>"]) for t in ctx_text.split()]
        tgt_ids = [self.word2idx.get(t.lower(), self.word2idx["<unk>"]) for t in tgt_text.split()]

        if len(ctx_ids) < self.max_len:
            ctx_ids += [self.word2idx["<pad>"]] * (self.max_len - len(ctx_ids))
        else:
            ctx_ids = ctx_ids[:self.max_len]

        if len(tgt_ids) < self.max_len:
            tgt_ids += [self.word2idx["<pad>"]] * (self.max_len - len(tgt_ids))
        else:
            tgt_ids = tgt_ids[:self.max_len]

        label = self.labels[idx]
        return torch.tensor(ctx_ids), torch.tensor(tgt_ids), torch.tensor(label)

def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_vocab(all_texts, sswe_emb, glove_emb, min_freq=1):
    """
    Build a vocabulary from the given texts.
    The vocabulary contains tokens found in either SSWE or GloVe embeddings.
    Tokens are added to the vocabulary only if their frequency exceeds min_freq.
    
    :param all_texts: List of texts from the dataset
    :param sswe_emb: Pre-trained SSWE embeddings (dictionary)
    :param glove_emb: Pre-trained GloVe embeddings (dictionary)
    :param min_freq: Minimum frequency for including tokens in the vocabulary
    :return: word2idx, idx2word - vocab mapping and reverse mapping
    """
    freq = {}
    for text in all_texts:
        for t in text.split():
            t = t.lower()
            freq[t] = freq.get(t, 0) + 1

    word2idx = {"<pad>": 0, "<unk>": 1}
    idx2word = ["<pad>", "<unk>"]
    for token, count in freq.items():
        if count >= min_freq and (token in sswe_emb or token in glove_emb):
            if token not in word2idx:
                word2idx[token] = len(idx2word)
                idx2word.append(token)
    return word2idx, idx2word

def get_dataloaders(batch_size=32, max_len=30, sswe_emb=None, glove_emb=None):
    dataset = load_dataset("oneonlee/cleansed_emocontext")
    def is_valid(example):
        return example["turn1"] is not None and example["turn2"] is not None and example["turn3"] is not None

    dataset["train"] = dataset["train"].filter(is_valid)
    dataset["validation"] = dataset["validation"].filter(is_valid)
    dataset["test"] = dataset["test"].filter(is_valid)

    train_turn1 = [clean_text(x["turn1"]) for x in dataset["train"]]
    train_turn2 = [clean_text(x["turn2"]) for x in dataset["train"]]
    train_turn3 = [clean_text(x["turn3"]) for x in dataset["train"]]
    train_labels = [x["label"] for x in dataset["train"]]

    val_turn1 = [clean_text(x["turn1"]) for x in dataset["validation"]]
    val_turn2 = [clean_text(x["turn2"]) for x in dataset["validation"]]
    val_turn3 = [clean_text(x["turn3"]) for x in dataset["validation"]]
    val_labels = [x["label"] for x in dataset["validation"]]

    test_turn1 = [clean_text(x["turn1"]) for x in dataset["test"]]
    test_turn2 = [clean_text(x["turn2"]) for x in dataset["test"]]
    test_turn3 = [clean_text(x["turn3"]) for x in dataset["test"]]
    test_labels = [x["label"] for x in dataset["test"]]

    all_texts = train_turn1 + train_turn2 + train_turn3 + val_turn1 + val_turn2 + val_turn3 + test_turn1 + test_turn2 + test_turn3
    word2idx, idx2word = build_vocab(all_texts, sswe_emb, glove_emb, min_freq=1)

    train_dataset = EmoContextDataset(train_turn1, train_turn2, train_turn3, train_labels, word2idx, max_len)
    val_dataset = EmoContextDataset(val_turn1, val_turn2, val_turn3, val_labels, word2idx, max_len)
    test_dataset = EmoContextDataset(test_turn1, test_turn2, test_turn3, test_labels, word2idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, word2idx, idx2word