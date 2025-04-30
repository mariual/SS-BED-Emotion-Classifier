import torch
import torch.nn as nn
from data_enhanced import get_dataloaders
from embeddings import load_sswe_embeddings, load_glove_embeddings, build_embedding_matrices
from model import SSBedContextModel
from train import train__enhanced_model, evaluate_enhanced_model
from evaluate import evaluate_and_print_enhanced_model
from data import get_dataloaders
from embeddings import load_sswe_embeddings, load_glove_embeddings, build_embedding_matrices
from model import SSBedModel
from train import train_model, evaluate_model
from evaluate import evaluate_and_print
import random
import numpy as np

def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Hyperparameters
    batch_size = 128
    max_len = 30
    hidden_dim = 64
    num_layers = 2
    sswe_dim = 50
    glove_dim = 100
    num_classes = 4
    dropout = 0.25
    learning_rate = 0.005
    epochs = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # File paths to your embedding text files
    sswe_path = "word_embeddings/sswe_embedding.txt"   # SSWE-u: 50D
    glove_path = "word_embeddings/glove.6B.100d.txt"   # GloVe: 100D

    print("Loading embeddings...")
    sswe_dict = load_sswe_embeddings(sswe_path)
    glove_dict = load_glove_embeddings(glove_path)

    print("Building DataLoaders...")
    # We pass the loaded dicts so we can build a combined vocabulary
    train_loader, val_loader, test_loader, word2idx, idx2word = get_dataloaders(
        batch_size=batch_size,
        max_len=max_len,
        sswe_emb=sswe_dict,
        glove_emb=glove_dict
    )

    print("Building embedding matrices...")
    sswe_matrix, glove_matrix = build_embedding_matrices(
        word2idx, 
        sswe_dict, 
        glove_dict,
        sswe_dim=sswe_dim,
        glove_dim=glove_dim
    )

    print("Initializing SS-BED model...")
    vocab_size = len(word2idx)
    model = SSBedModel(
        vocab_size=vocab_size,
        sswe_dim=sswe_dim,
        glove_dim=glove_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        sswe_matrix=sswe_matrix,
        glove_matrix=glove_matrix,
        dropout=dropout
    )

    # Train
    print("Training...")
    train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device)

    # Evaluate on test
    print("Final evaluation on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Print classification report (optional)
    evaluate_and_print(model, test_loader, device)

if __name__ == "__main__":
    main()


# # The Enhanced SS_BED model is not used in this script, but you can uncomment the following lines to use it.

# def main():

#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
#     # Hyperparameters
#     batch_size = 4000
#     max_len = 60
#     hidden_dim = 64
#     sswe_dim = 50
#     glove_dim = 100
#     num_classes = 4
#     dropout = 0.25
#     learning_rate = 0.005
#     epochs = 8
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     sswe_path = "word_embeddings/sswe_embedding.txt"
#     glove_path = "word_embeddings/glove.6B.100d.txt"

#     print("Loading embeddings...")
#     sswe_dict = load_sswe_embeddings(sswe_path)
#     glove_dict = load_glove_embeddings(glove_path)

#     print("Building DataLoaders...")
#     train_loader, val_loader, test_loader, word2idx, idx2word = get_dataloaders(batch_size, max_len, sswe_dict, glove_dict)

#     print("Building embedding matrices...")
#     sswe_matrix, glove_matrix = build_embedding_matrices(word2idx, sswe_dict, glove_dict, sswe_dim, glove_dim)

#     print("Initializing SS-BED Context Model...")
#     vocab_size = len(word2idx)
#     model = SSBedContextModel(vocab_size, sswe_dim, glove_dim, hidden_dim, num_classes, sswe_matrix, glove_matrix, dropout)

#     print("Training...")
#     train__enhanced_model(model, train_loader, val_loader, epochs, learning_rate, device)

#     print("Final evaluation on test set...")
#     criterion = nn.CrossEntropyLoss()
#     test_loss, test_acc = evaluate_enhanced_model(model, test_loader, criterion, device)
#     print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

#     evaluate_and_print_enhanced_model(model, test_loader, device)

# if __name__ == "__main__":
#     main()