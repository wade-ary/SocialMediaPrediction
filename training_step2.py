import math
import os
import torch
from torch.utils.data import DataLoader, Dataset

# --- your modules ---
from data_organisation import (
    build_master_train_table,
    materialize_tensors,
    train_data,
    val_data,
)
from training_step1 import SMPDataset          # expects keys x_cont, pair, y
from dcn import TwoHeadDCN

# ============== DATA HELPERS ==============
def make_ds_dl(tensor_info, *, batch_size=256, shuffle=False):
    x_meta, pair_emb, y = tensor_info["x_meta"], tensor_info["pair_emb"], tensor_info["y"]
    ds = SMPDataset(x_meta, pair_emb, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    return ds, dl

# ============== MAIN PIPELINE ==============
def main():
    # -------- Train build (fit artifacts on TRAIN only) --------
    master_train = build_master_train_table(train_data)

    # -------- Val build using TRAIN artifacts --------
    # ★ depends on your API:
    # If your build function accepts artifacts, pass them to the val build; else, use a dedicated eval builder if you have it.
    
    master_val = build_master_train_table(val_data)
    

    # -------- Materialize tensors (same layout for train/val) --------
    tensor_info_train = materialize_tensors(master_train)
    tensor_info_val   = materialize_tensors(master_val)

    # Sanity check on dims
    assert tensor_info_train["meta_cont_dim"] == tensor_info_val["meta_cont_dim"], "meta dims mismatch"
    assert tensor_info_train["emb_in_dim"]    == tensor_info_val["emb_in_dim"],    "emb dims mismatch"

    # -------- Datasets / Loaders --------
    train_ds, train_dl = make_ds_dl(tensor_info_train, batch_size=256, shuffle=True)
    val_ds,   val_dl   = make_ds_dl(tensor_info_val,   batch_size=512, shuffle=False)

    # -------- Model --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeadDCN(
        meta_cont_dim=tensor_info_train["meta_cont_dim"],   # e.g., 30
        emb_in_dim=tensor_info_train["emb_in_dim"],         # e.g., 1024
        meta_hidden=128,
        emb_hidden=256,
        cross_layers=3,
    ).to(device)

    # -------- Loss/Opt --------
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -------- Train with per-epoch val & immediate early stop --------
    os.makedirs("models", exist_ok=True)
    best_val = float("inf")
    epochs = 50  # can stop early anyway

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for batch in train_dl:
            x_cont = batch["x_cont"].to(device, non_blocking=True)  # (B, meta)
            pair   = batch["pair"].to(device, non_blocking=True)    # (B, emb)
            target = batch["y"].to(device, non_blocking=True)       # (B,)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(x_cont, pair)                          # (B,)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                              # unscale BEFORE clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * x_cont.size(0)
            n_train        += x_cont.size(0)

        train_mse  = train_loss_sum / max(n_train, 1)
        train_rmse = math.sqrt(train_mse)
        print(f"Epoch {epoch:02d} | train MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f}")

        # --- validation (no training; stop if no improvement) ---
        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for b in val_dl:
                x_val = b["x_cont"].to(device, non_blocking=True)
                p_val = b["pair"].to(device, non_blocking=True)
                y_val = b["y"].to(device, non_blocking=True)

                y_hat = model(x_val, p_val)
                l_val = criterion(y_hat, y_val)

                val_loss_sum += l_val.item() * x_val.size(0)
                n_val        += x_val.size(0)

        val_mse  = val_loss_sum / max(n_val, 1)
        val_rmse = math.sqrt(val_mse)
        print(f"              val MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f}")

        if epoch == 1:
            best_val = val_mse
            torch.save(model.state_dict(), "models/twohead_dcn.best.pt")
        else:
            if val_mse >= best_val - 1e-9:  # no improvement -> stop immediately
                print("No improvement on validation — stopping.")
                break
            best_val = val_mse
            torch.save(model.state_dict(), "models/twohead_dcn.best.pt")

    # (optional) save last epoch too
    torch.save(model.state_dict(), "models/twohead_dcn.last.pt")
    print(f"Done. Best val MSE: {best_val:.6f} (saved to models/twohead_dcn.best.pt)")

if __name__ == "__main__":
    main()
