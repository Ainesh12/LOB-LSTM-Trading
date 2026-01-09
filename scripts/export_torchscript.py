import os
import torch

from src.model import LOBLSTM 


def main():
    os.makedirs("artifacts", exist_ok=True)
    input_dim = 4
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2

    model = LOBLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    ckpt_path = "artifacts/model_state_dict.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    example = torch.zeros((1, 100, input_dim), dtype=torch.float32)

    ts = torch.jit.trace(model, example)
    out_path = "artifacts/model_ts.pt"
    ts.save(out_path)
    print("Saved TorchScript model to:", out_path)


if __name__ == "__main__":
    main()
