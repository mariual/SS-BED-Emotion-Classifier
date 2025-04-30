import torch
import torch.nn as nn

class SSBedModel(nn.Module):
    """
    SS-BED model from the paper:
      - Two parallel LSTMs:
         1) Sentiment-based word embeddings (SSWE)
         2) Semantic word embeddings (GloVe)
      - Their final hidden states are concatenated
      - Passed through a fully-connected layer to get output logits
    """

    def __init__(self, 
                 vocab_size, 
                 sswe_dim, 
                 glove_dim, 
                 hidden_dim,
                 num_layers,
                 num_classes,
                 sswe_matrix,
                 glove_matrix,
                 dropout=0.3):
        """
        :param vocab_size: size of vocabulary
        :param sswe_dim: dimension of SSWE embeddings
        :param glove_dim: dimension of GloVe embeddings
        :param hidden_dim: LSTM hidden size
        :param num_layers: # of LSTM layers
        :param num_classes: # of emotion classes (4 in this case)
        :param sswe_matrix: pre-initialized SSWE embedding matrix
        :param glove_matrix: pre-initialized GloVe embedding matrix
        :param dropout: dropout probability
        """

        super(SSBedModel, self).__init__()

        # Embedding layers
        self.sswe_embedding = nn.Embedding(vocab_size, sswe_dim, padding_idx=0)
        self.glove_embedding = nn.Embedding(vocab_size, glove_dim, padding_idx=0)

        # Initialize weights from pre-trained matrices
        self.sswe_embedding.weight.data.copy_(sswe_matrix)
        self.glove_embedding.weight.data.copy_(glove_matrix)

        # We do NOT necessarily require fine-tuning of these embeddings. 
        # If we want them trainable, we keep requires_grad=True. 
        # If not, we set them to False:
        self.sswe_embedding.weight.requires_grad = True
        self.glove_embedding.weight.requires_grad = True

        # Two LSTMs - one for SSWE, one for GloVe
        self.sswe_lstm = nn.LSTM(input_size=sswe_dim, 
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=False)
        
        self.glove_lstm = nn.LSTM(input_size=glove_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=False)

        # Intermediate fully connected layer to reduce dimensionality from 128 to 64
        self.fc1 = nn.Linear(2 * hidden_dim, 64)  # [B, 2*hidden_dim] -> [B, 64]
        
        # Final classification layer:
        self.fc = nn.Linear(64, num_classes)  # [B, 64] -> [B, num_classes]

        # Dropout and Leaky ReLU activation
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len] of token indices
        """
        # Get embeddings for both SSWE and GloVe
        sswe_emb = self.sswe_embedding(x)    # shape [B, L, sswe_dim]
        glove_emb = self.glove_embedding(x)  # shape [B, L, glove_dim]

        # Pass each through respective LSTM
        sswe_out, (sswe_h, sswe_c) = self.sswe_lstm(sswe_emb)
        glove_out, (glove_h, glove_c) = self.glove_lstm(glove_emb)

        # The final hidden states from the top layer of each LSTM
        sswe_final = sswe_h[-1]  # [B, hidden_dim]
        glove_final = glove_h[-1]  # [B, hidden_dim]

        # Concatenate the hidden states from both LSTMs
        combined = torch.cat((sswe_final, glove_final), dim=1)  # [B, 2*hidden_dim]

        # Apply dropout and Leaky ReLU
        combined = self.dropout(combined)
        combined = self.leaky_relu(self.fc1(combined))  # Reduced to 64

        # Final classification
        logits = self.fc(combined)  # [B, num_classes]

        return logits

    
# Context modeling model
class SSBedContextModel(nn.Module):
    def __init__(self, vocab_size, sswe_dim, glove_dim, hidden_dim, num_classes, sswe_matrix, glove_matrix, dropout=0.25):
        super().__init__()
        self.sswe_embedding = nn.Embedding.from_pretrained(sswe_matrix, freeze=False, padding_idx=0)
        self.glove_embedding = nn.Embedding.from_pretrained(glove_matrix, freeze=False, padding_idx=0)

        self.ctx_lstm_s = nn.LSTM(sswe_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ctx_lstm_g = nn.LSTM(glove_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.tgt_lstm_s = nn.LSTM(sswe_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.tgt_lstm_g = nn.LSTM(glove_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * 2 * hidden_dim, num_classes)

    def encode(self, x, embedding, lstm):
        x_embed = embedding(x)
        _, (h_n, _) = lstm(x_embed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # for BiLSTM
        return h

    def forward(self, ctx_tokens, tgt_tokens):
        ctx_s = self.encode(ctx_tokens, self.sswe_embedding, self.ctx_lstm_s)
        ctx_g = self.encode(ctx_tokens, self.glove_embedding, self.ctx_lstm_g)
        tgt_s = self.encode(tgt_tokens, self.sswe_embedding, self.tgt_lstm_s)
        tgt_g = self.encode(tgt_tokens, self.glove_embedding, self.tgt_lstm_g)

        combined = torch.cat([ctx_s, ctx_g, tgt_s, tgt_g], dim=1)
        return self.fc(self.dropout(combined))
