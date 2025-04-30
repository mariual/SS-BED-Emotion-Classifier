import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re

class EmoContextDataset(Dataset):
    """
    A PyTorch Dataset for single-utterance input:
    We only feed 'turn3' to the model and ignore turn1, turn2 for now.
    Each sample includes:
      - text (turn3)
      - label (numeric: 0=others, 1=happy, 2=sad, 3=angry)
    """

    def __init__(self, texts, labels, word2idx, max_len=30):
        """
        :param texts: list of raw text utterances (these come from turn3)
        :param labels: list of numeric labels [0,1,2,3]
        :param word2idx: dictionary mapping token -> index in vocabulary
        :param max_len: maximum sequence length for padding
        """
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]  # This is already numeric, e.g. 0,1,2,3

        # Tokenize (simple split). 
        tokens = text.split()

        # Convert tokens to IDs (lowercase them)
        token_ids = [self.word2idx.get(t.lower(), self.word2idx["<unk>"]) 
                     for t in tokens]

        # Pad/truncate
        if len(token_ids) < self.max_len:
            token_ids += [self.word2idx["<pad>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def normalize_emoticons(text):
    """
    Normalize emoticons to a canonical form for happy, sad, angry, and other emotions.
    This ensures that emoticons with different variants are treated consistently.
    """
    # Happy emoticons
    text = re.sub(r"(:-?[\)\]D\}]+|[\^0oO]+[\)_\]]*|:\]+)", ":)", text)  # :), :-), :D, :-D, ^_^, etc.
    
    # Sad emoticons
    text = re.sub(r"(:-?[\(\[{\|]+|[:;]?\'+\()", ":(", text)  # :(, :-(
    text = re.sub(r"T_T|Q_Q", ":(", text)  # crying face like T_T or Q_Q

    # Angry emoticons
    text = re.sub(r"(>[:\-]?[\(\|]+|>:[\-]?D|>@)", ">:(", text)  # >:(, >:-(
    text = re.sub(r"(:@|!@|>:(?:D|O))", ">:(", text)  # :@, >:O, >:D

    # Neutral / other faces
    text = re.sub(r"(:-[|]{2}|:-?o|:\/)", ":|", text)  # Neutral faces

    # For any unrecognized but consistent emoticons, map them to a default:
    text = re.sub(r"(:-?[\*\-]?X|=\\)", "<unk>", text)  # These don't fit the happy/sad/angry category.

    return text

def clean_text(text):
    """
    Basic text cleaning: 
    - lowercasing
    - remove extra spaces
    """
    text = normalize_emoticons(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_vocab(all_texts, sswe_emb, glove_emb, min_freq=1):
    """
    Build a vocabulary that can index every token found across the entire dataset.
    If a token isn't found in either embedding, we keep it but it will map to <unk>.
    """
    freq = {}
    for text in all_texts:
        tokens = text.split()
        for t in tokens:
            freq[t.lower()] = freq.get(t.lower(), 0) + 1

    # Start with special tokens
    word2idx = {"<pad>": 0, "<unk>": 1}
    idx2word = ["<pad>", "<unk>"]

    for token, count in freq.items():
        if count >= min_freq:
            # Only add if found in SSWE or GloVe
            if token in sswe_emb or token in glove_emb:
                if token not in word2idx:
                    word2idx[token] = len(idx2word)
                    idx2word.append(token)
    
    return word2idx, idx2word

def get_dataloaders(batch_size=32, max_len=30, sswe_emb=None, glove_emb=None):
    """
    Retrieves train, val, test DataLoaders for cleansed EmoContext data.
    We focus on single-utterance: only turn3.
    We assume the label column is numeric: 0=others,1=happy,2=sad,3=angry
    """

    # 1) Load dataset from huggingface
    dataset = load_dataset("oneonlee/cleansed_emocontext")

    # 2) Filter out rows where turn3 is None for each split
    dataset["train"] = dataset["train"].filter(lambda x: x["turn3"] is not None)
    dataset["validation"] = dataset["validation"].filter(lambda x: x["turn3"] is not None)
    dataset["test"] = dataset["test"].filter(lambda x: x["turn3"] is not None)

    # 3) Convert huggingface dataset into lists
    train_texts = [clean_text(x["turn3"]) for x in dataset["train"]]
    train_labels = [x["label"] for x in dataset["train"]]
    
    val_texts = [clean_text(x["turn3"]) for x in dataset["validation"]]
    val_labels = [x["label"] for x in dataset["validation"]]

    test_texts = [clean_text(x["turn3"]) for x in dataset["test"]]
    test_labels = [x["label"] for x in dataset["test"]]

    # 4) Build vocab from all texts
    all_texts = train_texts + val_texts + test_texts
    word2idx, idx2word = build_vocab(all_texts, sswe_emb, glove_emb, min_freq=1)

    # 5) Create Datasets
    train_dataset = EmoContextDataset(train_texts, train_labels, word2idx, max_len)
    val_dataset = EmoContextDataset(val_texts, val_labels, word2idx, max_len)
    test_dataset = EmoContextDataset(test_texts, test_labels, word2idx, max_len)

    # 6) Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, word2idx, idx2word




# # ---------------------------------------------------------------------------------------------------
# # here we use Turn1 and Turn2 as well (flattened dialogue context)
# # ---------------------------------------------------------------------------------------------------
# import torch
# from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset
# import re
# import string

# class EmoContextDataset(Dataset):
#     """
#     A PyTorch Dataset for single-utterance input:
#     We only feed 'turn3' to the model and ignore turn1, turn2 for now.
#     Each sample includes:
#       - text (turn3)
#       - label (numeric: 0=others, 1=happy, 2=sad, 3=angry)
#     """

#     def __init__(self, turn1s, turn2s, turn3s, labels, word2idx, max_len=60):
#         self.turn1s = turn1s
#         self.turn2s = turn2s
#         self.turn3s = turn3s
#         self.labels = labels
#         self.word2idx = word2idx
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.turn3s)

#     def __getitem__(self, idx):
#         # Combine turn1, turn2, turn3 into a single context string
#         turn1 = self.turn1s[idx] if self.turn1s[idx] is not None else ""
#         turn2 = self.turn2s[idx] if self.turn2s[idx] is not None else ""
#         turn3 = self.turn3s[idx]

#         full_text = clean_text(turn1 + " " + turn2 + " " + turn3)

#         tokens = full_text.split()
#         token_ids = [self.word2idx.get(t.lower(), self.word2idx["<unk>"]) for t in tokens]

#         # Pad/truncate
#         if len(token_ids) < self.max_len:
#             token_ids += [self.word2idx["<pad>"]] * (self.max_len - len(token_ids))
#         else:
#             token_ids = token_ids[:self.max_len]

#         label = self.labels[idx]
#         return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# def clean_text(text):
#     """
#     Basic text cleaning: 
#     - lowercasing
#     - remove extra spaces
#     """
#     text = text.lower()
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# def build_vocab(all_texts, sswe_emb, glove_emb, min_freq=1):
#     """
#     Build a vocabulary that can index every token found across the entire dataset.
#     If a token isn't found in either embedding, we keep it but it will map to <unk>.
#     """
#     freq = {}
#     for text in all_texts:
#         tokens = text.split()
#         for t in tokens:
#             freq[t.lower()] = freq.get(t.lower(), 0) + 1

#     # Start with special tokens
#     word2idx = {"<pad>": 0, "<unk>": 1}
#     idx2word = ["<pad>", "<unk>"]

#     for token, count in freq.items():
#         if count >= min_freq:
#             # Only add if found in SSWE or GloVe
#             if token in sswe_emb or token in glove_emb:
#                 if token not in word2idx:
#                     word2idx[token] = len(idx2word)
#                     idx2word.append(token)
    
#     return word2idx, idx2word

# def get_dataloaders(batch_size=32, max_len=30, sswe_emb=None, glove_emb=None):
#     """
#     Retrieves train, val, test DataLoaders for cleansed EmoContext data.
#     We focus on single-utterance: only turn3.
#     We assume the label column is numeric: 0=others,1=happy,2=sad,3=angry
#     """

#     # 1) Load dataset from huggingface
#     dataset = load_dataset("oneonlee/cleansed_emocontext")

#     # 2) Filter out rows where turns are None for each split
#     def is_valid(example):
#         return (
#             example["turn1"] is not None and
#             example["turn2"] is not None and
#             example["turn3"] is not None
#         )

#     dataset["train"] = dataset["train"].filter(is_valid)
#     dataset["validation"] = dataset["validation"].filter(is_valid)
#     dataset["test"] = dataset["test"].filter(is_valid)



#     train_turn1 = [clean_text(x["turn1"]) for x in dataset["train"]]
#     train_turn2 = [clean_text(x["turn2"]) for x in dataset["train"]]
#     train_turn3 = [clean_text(x["turn3"]) for x in dataset["train"]]
#     train_labels = [x["label"] for x in dataset["train"]]

#     val_turn1 = [clean_text(x["turn1"]) for x in dataset["validation"]]
#     val_turn2 = [clean_text(x["turn2"]) for x in dataset["validation"]]
#     val_turn3 = [clean_text(x["turn3"]) for x in dataset["validation"]]
#     val_labels = [x["label"] for x in dataset["validation"]]

#     test_turn1 = [clean_text(x["turn1"]) for x in dataset["test"]]
#     test_turn2 = [clean_text(x["turn2"]) for x in dataset["test"]]
#     test_turn3 = [clean_text(x["turn3"]) for x in dataset["test"]]
#     test_labels = [x["label"] for x in dataset["test"]]


#     # 4) Build vocab from all texts
#     all_texts = train_turn1 + train_turn2 + train_turn3 + \
#                 val_turn1 + val_turn2 + val_turn3 + \
#                 test_turn1 + test_turn2 + test_turn3
#     word2idx, idx2word = build_vocab(all_texts, sswe_emb, glove_emb, min_freq=1)

#     # 5) Create Datasets
#     train_dataset = EmoContextDataset(train_turn1, train_turn2, train_turn3, train_labels, word2idx, max_len)
#     val_dataset = EmoContextDataset(val_turn1, val_turn2, val_turn3, val_labels, word2idx, max_len)
#     test_dataset = EmoContextDataset(test_turn1, test_turn2, test_turn3, test_labels, word2idx, max_len)


#     # 6) Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader, word2idx, idx2word