import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from gsfl.data import get_test_loader
from gsfl.models import ClientNet, ServerNet

# Auto-add project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)


def evaluate_and_get_preds(client, server, test_loader):
    client.eval()
    server.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to("cpu")
            
            smashed = client(images)
            outputs = server(smashed)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    classes = [str(i) for i in range(10)]
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()


def load_model(client_path, server_path):
    client = ClientNet()
    server = ServerNet()

    client.load_state_dict(torch.load(client_path, map_location="cpu"))
    server.load_state_dict(torch.load(server_path, map_location="cpu"))

    return client, server


def main():
    test_loader = get_test_loader()

    print("Loading saved models for SL and GSFL...")

    # Expected model paths (you can rename them)
    SL_CLIENT = "results/sl_client.pt"
    SL_SERVER = "results/sl_server.pt"
    GSFL_CLIENT = "results/gsfl_client.pt"
    GSFL_SERVER = "results/gsfl_server.pt"

    if not all(os.path.exists(p) for p in [SL_CLIENT, SL_SERVER, GSFL_CLIENT, GSFL_SERVER]):
        print("❌ ERROR: Model files missing! Please save SL and GSFL models first.")
        return

    # Load models
    sl_client, sl_server = load_model(SL_CLIENT, SL_SERVER)
    gsfl_client, gsfl_server = load_model(GSFL_CLIENT, GSFL_SERVER)

    print("Evaluating SL...")
    y_true_sl, y_pred_sl = evaluate_and_get_preds(sl_client, sl_server, test_loader)
    plot_confusion_matrix(y_true_sl, y_pred_sl, "Confusion Matrix - SL", "cm_sl.png")

    print("Evaluating GSFL...")
    y_true_gsfl, y_pred_gsfl = evaluate_and_get_preds(gsfl_client, gsfl_server, test_loader)
    plot_confusion_matrix(y_true_gsfl, y_pred_gsfl, "Confusion Matrix - GSFL", "cm_gsfl.png")

    print("🎉 Confusion matrices saved inside results/:")
    print(" - results/cm_sl.png")
    print(" - results/cm_gsfl.png")


if __name__ == "__main__":
    main()
