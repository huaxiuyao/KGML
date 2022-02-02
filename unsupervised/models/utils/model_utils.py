
import torch

def save_model(model, ckpt_dir, epoch, device):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{epoch}_weights.pkl'
    model_state = model.to('cpu').state_dict(),

    # Overwriste best_weights.pkl with the latest.
    torch.save(model_state, ckpt_path)
    ckpt_path = ckpt_dir / f'best_weights.pkl'
    torch.save(model_state, ckpt_path)
    model.to(device)
