import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---- import your data pipeline bits ----
# Assumes these names are in your data_organisation.py exactly as you showed.
from data_organisation import (
    build_master_train_table,
    materialize_tensors,
    train_data,   # this is the dict with train splits you constructed earlier
)


# ------------- Dataset wrapper -------------
class SMPDataset(Dataset):
    def __init__(self, x_meta: np.ndarray, pair_emb: np.ndarray, y: np.ndarray):
        assert x_meta.ndim == 2, f"x_meta must be (N, D), got {x_meta.shape}"
        assert pair_emb.ndim == 2, f"pair_emb must be (N, E), got {pair_emb.shape}"
        assert y.ndim == 1, f"y must be (N,), got {y.shape}"
        assert x_meta.shape[0] == pair_emb.shape[0] == y.shape[0], "N mismatch"

        self.x_meta  = torch.from_numpy(x_meta).float()
        self.pair_emb= torch.from_numpy(pair_emb).float()
        self.y       = torch.from_numpy(y).float()

    def __len__(self):
        return self.x_meta.shape[0]

    def __getitem__(self, idx):
        return {
            "x_cont": self.x_meta[idx],     # (D_meta,)
                   # you already one-hot encoded; keep None
            "pair":   self.pair_emb[idx],   # (emb_in_dim,)
            "y":      self.y[idx],          # scalar target
        }

def main():
    print("▶ Building master training table...")
    master = build_master_train_table(train_data)

    print("▶ Materializing tensors...")
    tensor_info = materialize_tensors(master)

    x_meta   = tensor_info["x_meta"]      # (N, 30)
    pair_emb = tensor_info["pair_emb"]    # (N, 1024)  text(512)+video(512)
    y        = tensor_info["y"]           # (N,)

    print(f"Shapes -> x_meta: {x_meta.shape}, pair_emb: {pair_emb.shape}, y: {y.shape}")

    ds = SMPDataset(x_meta, pair_emb, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    # pull one sanity batch
    batch = next(iter(dl))
    print("One batch:")
    print(" x_cont  :", batch["x_cont"].shape)   # (B, 30)
    print(" x_cats  :", batch["x_cats"])         # None
    print(" pair    :", batch["pair"].shape)     # (B, 1024)
    print(" y       :", batch["y"].shape)        # (B,)

if __name__ == "__main__":
    main()