from sklearn.metrics import precision_recall_fscore_support
import torch

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        y_true.append(labels.cpu())
        y_pred.append(probs.cpu())

    # Concatenar todos los batches
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # 🔥 APLICAR UMBRAL para binarizar
    y_pred_bin = (y_pred >= threshold).astype(int)

    num_classes = y_true.shape[1]
    precisions, recalls, f1s, supports = [], [], [], []

    for i in range(num_classes):
        p, r, f1, sup = precision_recall_fscore_support(
            y_true[:, i], y_pred_bin[:, i],
            average='binary', zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(sup)

    uar = sum(recalls) / num_classes
    uf1 = sum(f1s) / num_classes

    metrics = {
        "avg_loss": total_loss / len(dataloader),
        "uar": uar,
        "uf1": uf1,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "supports": supports
    }
    return total_loss, metrics, (y_true, y_pred_bin)