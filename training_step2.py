import torch
from torch.utils.data import DataLoader
import numpy as np

# your bits from Step 1
from data_organisation import build_master_train_table, materialize_tensors, train_data
from training_step1 import SMPDataset

# 🔹 import your model architecture (the file where TwoHeadDCN lives)
from dcn import TwoHeadDCN   # <-- change to your filename

def main():
    # -------- data --------
    master = build_master_train_table(train_data)
    tensor_info = materialize_tensors(master)
    x_meta, pair_emb, y = tensor_info["x_meta"], tensor_info["pair_emb"], tensor_info["y"]

    ds = SMPDataset(x_meta, pair_emb, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    # -------- model --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeadDCN(
        meta_cont_dim=tensor_info["meta_cont_dim"],   # 30                       # you already one-hot'd cats
        emb_in_dim=tensor_info["emb_in_dim"],         # 1024
        meta_hidden=128,
        emb_hidden=256,
        cross_layers=3,
    ).to(device)

    # -------- loss/opt --------
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -------- train --------
    model.train()
    epochs = 5
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in dl:
            x_cont = batch["x_cont"].to(device, non_blocking=True)     # (B, 30)                                           # (no cat embs)
            pair   = batch["pair"].to(device, non_blocking=True)       # (B, 1024)
            target = batch["y"].to(device, non_blocking=True)          # (B,)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(x_cont,pair)                     # (B,)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x_cont.size(0)

        epoch_loss = running_loss / len(ds)
        print(f"Epoch {epoch:02d} | train MSE: {epoch_loss:.4f} | RMSE: {epoch_loss**0.5:.4f}")

    # (optional) save the model
    torch.save(model.state_dict(), "twohead_dcn.pt")
    print("Saved model to twohead_dcn.pt")

if __name__ == "__main__":
    main()