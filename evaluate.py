from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def evaluate_and_print(model, data_loader, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            outputs = model(token_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Full report including all 4 classes ---
    print("Full Classification Report (including 'others'):")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=["others", "happy", "sad", "angry"]
    ))

    # --- Focused metrics: only emotion classes (1, 2, 3) ---
    labels_emotions = [1, 2, 3]
    print("\nEmotion-only Macro/Micro F1 (excluding 'others'):")
    macro_f1 = f1_score(all_labels, all_preds, labels=labels_emotions, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, labels=labels_emotions, average='micro')
    print(f"Macro F1 (happy/sad/angry): {macro_f1:.4f}")
    print(f"Micro F1 (happy/sad/angry): {micro_f1:.4f}")

    # --- Confusion Matrix ---
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["others", "happy", "sad", "angry"],
                yticklabels=["others", "happy", "sad", "angry"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()




def evaluate_and_print_enhanced_model(model, data_loader, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ctx, tgt, labels in data_loader:
            ctx, tgt, labels = ctx.to(device), tgt.to(device), labels.to(device)
            outputs = model(ctx, tgt)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Full Classification Report (including 'others'):")
    print(classification_report(all_labels, all_preds, target_names=["others", "happy", "sad", "angry"]))

    print("\nEmotion-only Macro/Micro F1 (excluding 'others'):")
    macro_f1 = f1_score(all_labels, all_preds, labels=[1, 2, 3], average='macro')
    micro_f1 = f1_score(all_labels, all_preds, labels=[1, 2, 3], average='micro')
    print(f"Macro F1 (happy/sad/angry): {macro_f1:.4f}")
    print(f"Micro F1 (happy/sad/angry): {micro_f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["others", "happy", "sad", "angry"], yticklabels=["others", "happy", "sad", "angry"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()