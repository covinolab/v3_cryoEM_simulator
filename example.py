"""Render ten TRPV1 particle images with v3 and save them to example_images.npy."""
from pathlib import Path

import numpy as np
import torch

from v3_core import DEFAULT_PRIOR, RenderShape, load_bank_pt, pack_phi, render, sample_theta

DATA = Path(__file__).resolve().parent / "data"
device = "cuda" if torch.cuda.is_available() else "cpu"

bank = load_bank_pt(DATA / "trpv1_bank.pt", device=device)       # 4 TRPV1 conformers, one bead per residue
shape = RenderShape(num_pixels=192, pixel_size=1.345,             # the TRPV1 particles: 192 px of 1.345 A
                    noise={"empirical_cache_file": str(DATA / "trpv1_psd.pt")})   # their noise spectrum

phi = pack_phi(DEFAULT_PRIOR, device=device)                      # the prior: 44 numbers
theta = sample_theta(phi, 10, seed=0, bank=bank, cfg=shape)       # the settings of 10 images: (10, 156)
images = render(theta, bank, shape)                               # the images: (10, 192, 192)

np.save("example_images.npy", images.cpu().numpy())
print(f"rendered {tuple(images.shape)} on {device} -> example_images.npy")
