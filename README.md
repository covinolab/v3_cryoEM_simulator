# v3: a simulator of cryo-EM particle images

v3 turns the 3-D structure of a molecule into **synthetic cryo-EM particle images** that resemble those of an **experimental** single-particle data set: the molecule seen from random directions, surrounded by neighbouring objects, blurred by the microscope, buried in noise, and sometimes cut off at the micrograph edge. Everything is in one file, `v3_core.py`, written in PyTorch; it runs on a CPU or a GPU and is differentiable.


## Installation

```bash
mkdir v3_simulator
cd v3_simulator
git clone https://github.com/covinolab/v3_cryoEM_simulator.git
cd v3_cryoEM_simulator
pip install -r requirements.txt       # PyTorch, NumPy, MDAnalysis, mrcfile, matplotlib
pip install jupyter                   # only for the tutorial
```

Python 3.10 or newer and PyTorch 2.0 or newer. Tested with Python 3.12 and PyTorch 2.8, on CPU and CUDA. MDAnalysis is needed only to read PDB files, mrcfile only to read particle files.

## Quick start

```python
import v3_core as v3

bank   = v3.load_bank("data/pdb")                                # the molecule: one conformer per PDB file
shape  = v3.RenderShape(num_pixels=192, pixel_size=1.345)        # the box: 192 pixels of 1.345 Angstrom
phi    = v3.pack_phi(v3.DEFAULT_PRIOR)                           # the prior: 44 numbers
theta  = v3.sample_theta(phi, 10, seed=0, bank=bank, cfg=shape)  # the settings of 10 images: (10, 156)
images = v3.render(theta, bank, shape)                           # the images: (10, 192, 192)
```

* **`tutorial.ipynb`** explains every step. Part 1 starts from the single PDB file `data/pdb/5irx.pdb` and makes ten images with the default prior; Part 2 adds a conformer bank, the noise spectrum of an experimental data set and a prior of your own. Start here if you have never used v3.
* **`example.py`** renders ten images with the four-conformer TRPV1 bank and the measured TRPV1 noise spectrum, and saves them to `example_images.npy`: `python example.py`.

## How an image is made

v3 keeps three objects apart:

| object | size | what it describes | example |
|---|---|---|---|
| **φ**, the prior | 44 numbers | how the settings vary across a whole data set | "the defocus lies between 0.06 and 5.6 µm" |
| **θ**, the settings | 156 numbers per image | the settings of one image, drawn at random from φ | "this image has a defocus of 1.7 µm" |
| **image** | N × N pixels | the picture made from θ | |

`sample_theta(phi, n, seed, bank, shape)` draws θ; `render(theta, bank, shape)` makes the images. The **bank** holds the 3-D structures; the **shape** (`RenderShape`) holds the box size, the pixel size and the source of the noise spectrum. `render` makes each image in five stages:

1. **Projection.** The molecule is rotated to the image's viewing direction; a Gaussian bead of width σ is placed on every point of the model (one point per residue), and the beads are added up along the viewing direction.
2. **Clutter.** Zero to five objects are added around the molecule: whole copies of it, compact pieces holding a fraction of its residues, and random point clouds standing in for debris, detergent or ice contamination. The point clouds mix five basic shapes (compact, shell, filament, branched, sheet). Each object sits at a random angle and at a distance r = r_out · u^(1/(2a)) from the box centre (u uniform, r_out half the box diagonal, a = `radial_exponent`), so some land partly or entirely outside the box.
3. **Microscope blur (CTF).** In Fourier space every spatial frequency f is multiplied by

   C(f) = −(1/a) · [a cos χ + √(1 − a²) sin χ] · exp(−B |f|² / 2),  χ = π λ Δz |f|²,

   with amplitude contrast a = 0.1, electron wavelength λ = 0.019866 Å, defocus Δz and B-factor B. The image is zero-padded before the Fourier transform, so that blurred signal does not wrap around the box edges.
4. **Noise.** Three noise fields of variance 1 are mixed with the image's weights, rescaled to variance 1 again, and added with the strength σₙ set by the SNR:

   noisy image = clean image + σₙ · normalise(w_col · coloured + w_shot · shot + w_white · white).

   *Coloured* noise is white noise shaped by the 2-D power spectrum of experimental particles; *shot* noise imitates counting electrons, with a dose λ = clip(dose · SNR, 1, 200); *white* noise is the same at every spatial scale. Each image gets coloured noise only, white noise only, or a mix of the three.
5. **Edge band and standardisation.** When a particle lies at the border of its micrograph, extraction software fills the missing part of the box by repeating the last row (or column) of pixels. v3 reproduces this band in a fraction of the images. Finally every image is shifted and scaled to mean 0 and standard deviation 1.

**The noise spectrum** comes from the first source that is available: (1) the spectrum of your data set, given to `RenderShape` as a file or measured from a folder of particle files; (2) the average spectrum of several experimental data sets in `data/global_average_empirical_psd.pt`; (3) a simple 1/r^0.39 formula. The fallback is silent, so check that your spectrum file is found.

## The prior φ

`v3.DEFAULT_PRIOR` is a dictionary of physical values, the default ones. A range `[lo, hi]` means that each image draws a value uniformly between lo and hi; a single number between 0 and 1 is a probability. `pack_phi` turns the dictionary into the 44 stored numbers, and `unpack_phi` turns them back. To use a different prior, copy the dictionary, change entries, and pack it again.

| group | entry | default | meaning |
|---|---|---|---|
| optics | `SIGMA` | [0.261, 2.41] | bead width σ, Å |
| | `DEFOCUS` | [0.0551, 5.62] | defocus, µm |
| | `B_FACTOR` | [0.0871, 129.0] | B-factor, Å² |
| | `SNR` | [-5.8, -1.5] | log10 of the signal-to-noise ratio |
| noise | `MIX_RANGE_gaussian` | [0.0, 0.784] | weight of the white noise |
| | `MIX_RANGE_fft` | [0.0, 0.956] | weight of the coloured noise |
| | `MIX_RANGE_poisson` | [0.0, 0.498] | weight of the shot noise |
| | `CONC_fft` | 8.49 | pushes the coloured-noise weight towards the top of its range: w = lo + (hi − lo) · u^(1/CONC_fft) |
| | `WHITE_FRAC_fft` | 0.205 | probability that the coloured-noise weight comes from the bottom quarter of its range instead (mostly white images) |
| | `REGIME_full_psd` | 0.392 | probability of coloured noise only |
| | `REGIME_white` | 0.151 | probability of white noise only (never below 0.02; the share is min(REGIME_white, 1 − REGIME_full_psd)) |
| clutter | `count_probs` | [0.458, 0.153, 0.134, 0.118, 0.085, 0.051] | probabilities of 0, 1, 2, 3, 4, 5 objects |
| | `p_bank` | 0.672 | probability that an object contains part of the molecule; otherwise it is a random point cloud |
| | `p_duplicate` | 0.391 | probability that an object is a whole copy of the molecule |
| | `p_free_amp` | 0.373 | probability that a copy or piece gets a freely drawn brightness instead of one set by its number of residues |
| | `rg_range_A` | [1.2, 137.0] | size (radius of gyration) of the point clouds, Å |
| | `contrast_range` | [-0.92, 1.41] | brightness of the point clouds; negative values make holes |
| | `amp_free_range` | [0.0, 1.14] | the freely drawn brightness of copies and pieces |
| | `aniso_log_halfwidth` | 1.59 | how much point clouds are stretched: each log-stretch is uniform in [−value, +value] |
| | `radial_exponent` | 1.26 | a in the placement rule of stage 2; a > 1 pushes objects outwards |
| band | `barcode_prob` | 0.139 | probability of an edge band |
| | `barcode_width_frac` | [0.0, 0.292] | band depth, as a fraction of the box size |
| | `side_weights` | [0.239, 0.305, 0.238, 0.219] | probabilities of the top, bottom, left and right edge |
| | `depth_a`, `depth_b` | 0.913, 1.19 | shape of the depth distribution, a Kumaraswamy(a, b); 1 and 1 is uniform |

`count_probs` and `side_weights` are divided by their sums. Ranges need `lo < hi`, and the noise weights stay within [0, 1].

## The settings θ

Each image has 156 numbers. `unpack_theta(theta)` returns them as a dictionary of named pieces and `pack_theta` joins them back, so you can edit θ and render again (the tutorial does this). `v3.THETA_LAYOUT` describes every entry.

| group | numbers | entries |
|---|---|---|
| seed | 1 | `seed`: regenerates the image's fixed random point clouds |
| base | 13 | `conformer_idx` (2: conformer, representative), `quaternion` (4: viewing direction), `sigma`, `shift` (2; 0 unless `RenderShape.shift_A` is set), `defocus`, `b_factor`, `amp` (0.1), `log10_snr` |
| noise | 5 | `noise_w` (3: weights of coloured, shot and white noise), `noise_regime` (0 coloured only, 1 white only, 2 mixed), `noise_white_flag` |
| band | 7 | `band_present`, `band_width_frac`, `band_side` (0 top, 1 bottom, 2 left, 3 right), `band_side_onehot` (4) |
| clutter | 130 | 5 slots × 26 numbers: presence, type, piece position, conformer, brightness, shape weights and shape numbers, size, stretch, orientation, position |

## Your own molecule and data

**The molecule.** `v3.load_bank(folder)` reads every PDB file in a folder, centres it, and keeps one point per residue: C-alpha atoms by default, or `atom_selection="name P"` for nucleic acids. Each file becomes one conformer, read in alphabetical order; images draw conformers with equal probability, and files with fewer residues are padded. A bank saved with `torch.save` is read with `v3.load_bank_pt(file)`. The bank has the shape (conformers, representatives, 3, points), in Å.

**The box.** Give `RenderShape` the box size and pixel size of the experimental particles you want to imitate: `v3.RenderShape(num_pixels=N, pixel_size=p)`.

**The noise spectrum.** Measure it on experimental particles of the same box size, save it, and point `RenderShape` to the file:

```python
stats = v3.estimate_empirical_noise_parameters("my_particles/", num_images=10000)   # a folder of .mrc / .mrcs files
torch.save(stats, "my_psd.pt")
shape = v3.RenderShape(num_pixels=N, pixel_size=p, noise={"empirical_cache_file": "my_psd.pt"})
```

`noise={"empirical_root_folder": "my_particles/"}` measures the spectrum when the first image is rendered instead. The pixel size is not read from the particle files.

## Data in this repository

| path | content | source |
|---|---|---|
| `data/pdb/5irx.pdb` | TRPV1 with DkTx and RTX in a lipid nanodisc; the tutorial's starting structure | PDB [5IRX](https://www.rcsb.org/structure/5IRX); Gao, Cao, Julius and Cheng, *Nature* 534, 347 (2016) |
| `data/trpv1_particles/` | 100 experimental TRPV1 particles, 192 × 192 pixels of 1.345 Å; `SOURCE.txt` lists each one's origin | [EMPIAR-10059](https://www.ebi.ac.uk/empiar/EMPIAR-10059/), same study |
| `data/trpv1_psd.pt` | the noise spectrum of those 100 particles (made by the tutorial) | |
| `data/trpv1_bank.pt` | four TRPV1 conformers (toxin-bound, toxin-faded, closed apo-like, open ligand-bound), one point per residue | |
| `data/global_average_empirical_psd.pt` | the average noise spectrum of eight experimental particle sets, used when no spectrum is given | |

PDB and EMPIAR entries are released under CC0.

## Citation

If you use v3, please cite the paper (the metadata are in `CITATION.cff`):


## License

MIT, see `LICENSE`. The projection, CTF and white noise building blocks of `v3_core.py` come from [cryoSBI](https://github.com/flatironinstitute/cryoSBI).
