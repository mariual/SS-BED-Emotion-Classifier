import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device="cuda"):
    """
    Train loop for SS-BED model.
    :param model: instance of SSBedModel
    :param train_loader: DataLoader for training
    :param val_loader: DataLoader for validation
    :param epochs: number of epochs
    :param lr: learning rate
    :param device: "cuda" or "cpu"
    """

    model.to(device)

    # Cross Entropy Loss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for batch_idx, (token_ids, labels) in enumerate(train_loader):
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(token_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # Evaluate on validation after each epoch
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.4f}")

def evaluate_model(model, data_loader, criterion, device="cuda"):
    """
    Evaluate model on a given DataLoader. 
    Returns (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            outputs = model(token_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # predictions
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy



# train steps for training the enhanced SS-BED model
def train__enhanced_model(model, train_loader, val_loader, epochs=10, lr=0.005, device="cuda"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for ctx, tgt, label in train_loader:
            ctx, tgt, label = ctx.to(device), tgt.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(ctx, tgt)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_enhanced_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

def evaluate_enhanced_model(model, data_loader, criterion, device="cuda"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ctx, tgt, label in data_loader:
            ctx, tgt, label = ctx.to(device), tgt.to(device), label.to(device)
            outputs = model(ctx, tgt)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    return total_loss / len(data_loader), correct / total
