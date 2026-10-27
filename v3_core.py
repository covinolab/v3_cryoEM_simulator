"""
v3_core.py -- a fast, differentiable simulator of cryo-EM particle images.
The paper calls this simulator "v3".

This is the simulator of the paper

    A. Zafranas and R. Covino,
    "A quantitative approach to characterise model misspecification in cryo-EM",
    NeurIPS 2026 workshop paper.

In the comments, "the paper" means that article, and "SI" means its supplementary
information. Eq. S7 ... S17, Tables S2-S3 and Figs. 2-3 are referred to by those
numbers.

Everything is in this one file. You need Python 3.10 or newer and PyTorch 2.0 or
newer (tested with 2.14). Two extras are optional: `mrcfile` (only to measure a
noise spectrum from your own particle files) and `MDAnalysis` (only to read
molecules from PDB files). `example.py`, next to this file, renders ten images;
`tutorial.ipynb` goes through every step.


===============================================================================
WHAT THIS SIMULATOR DOES
===============================================================================
In cryo-electron microscopy (cryo-EM), many copies of a molecule are frozen in a
thin layer of ice and photographed with electrons. Each copy appears as a small,
very noisy, grey picture called a "particle image", seen from an unknown
direction. This file makes synthetic particle images.

It starts from the 3-D coordinates of a molecule (one point per amino acid, or
per nucleotide for nucleic acids). It turns them into a flat picture, adds neighbouring
molecules and debris, blurs everything the way the microscope does, adds
realistic noise, sometimes adds a band of repeated pixel lines that experimental data
sets contain along one edge, and finally rescales the picture.

Every random choice comes from a short list of settings, the prior phi below.
Almost every step is written so that PyTorch can take derivatives of the images
with respect to those settings.


===============================================================================
THE THREE OBJECTS:  phi  ->  theta  ->  image
===============================================================================
phi   (44 numbers)   The PRIOR: how images vary across a whole data set, for
                     example "the defocus is between 0.5 and 3.5 micrometres".
                     The paper's Table S3 lists all 44 entries.
theta (156 numbers)  The settings of ONE image, drawn at random from phi, for
                     example "this image has a defocus of 1.7 micrometres".
                     Listed at the bottom of the paper's Table S2 (Eq. S13).
image (N x N pixels) The picture made from theta.

    phi    = pack_phi(DEFAULT_PRIOR)                     # 44 numbers
    theta  = sample_theta(phi, n, seed, bank, shape)     # n rows of 156 numbers
    images = render(theta, bank, shape)                  # n pictures, N x N

`bank` holds the 3-D structures (conformers) to draw from. `shape` (a
RenderShape) holds the box size, the pixel size and where the noise spectrum
comes from.


===============================================================================
HOW ONE IMAGE IS MADE  (the paper's SI S3 and Fig. 2)
===============================================================================
Stage 1  Projection (Eq. S14). Rotate the molecule, put a small Gaussian blob
         ("bead") on every atom kept in the model, and add the blobs up along
         the viewing direction.
         Before anything else is added, the clean molecule alone is passed
         through the microscope blur (Stage 3), and the spread (standard
         deviation) of its pixel values inside the circle that touches the
         box edges is measured. This fixes the noise level later (Eq. S15), so
         the signal-to-noise ratio refers to the molecule, not to the clutter
         around it.
Stage 2  Clutter (Eq. S7-S9). Add 0 to 5 extra objects around the molecule:
         copies or pieces of the same molecule, and random point clouds that
         stand in for debris, detergent or ice contamination.
Stage 3  Microscope blur, the "contrast transfer function" or CTF (Eq. S16),
         applied to the molecule and its clutter together.
Stage 4  Noise (Eq. S10, S11, S17). A mix of white noise, coloured noise with
         the spectrum measured on experimental particles, and shot noise.
Stage 5  Edge band (Eq. S12), then standardisation. Sometimes fill a band along
         one edge with copies of the image line next to it, as real extraction
         software does when a particle sits at the border of the micrograph.
         Finally shift and scale the image to mean 0 and standard deviation 1.


===============================================================================
WORDS USED IN THIS FILE
===============================================================================
amplitude contrast  The small part of the image contrast (a = 0.1 here, Eq. S16)
                    that does not come from the phase shift of the electrons.
                    Without it, the coarsest features would vanish.
Angstrom (A)        0.1 nanometre. Distances on the molecule are in Angstrom.
bank                The set of 3-D structures (conformers) images are drawn from.
batch               Several images made together in one call.
bead                The small round Gaussian blob drawn for each point of the
                    molecule. Its width sigma is the Gaussian's standard
                    deviation, in Angstrom (the full width at half height is
                    2.355 sigma).
B-factor            How strongly fine details fade (a blur strength), in A^2.
                    Eq. S16 writes the fading as exp(-B |f|^2 / 2). Many cryo-EM
                    programs write exp(-B |f|^2 / 4), so the same blur has twice
                    the B there.
coloured noise      Noise that is stronger at some spatial scales than others,
                    as in experimental micrographs. The opposite is "white" noise.
conformer           One 3-D shape of the molecule. A flexible molecule has many.
CTF                 Contrast transfer function: how the microscope turns a
                    sharp projection into a blurred, ringing image.
defocus             How far from focus the microscope is, in micrometres. More
                    defocus means stronger, wider rings around features.
FFT order           How torch.fft stores frequencies: zero frequency at index
                    [0, 0], then the positive frequencies, then the negative
                    ones. torch.fft.fftshift moves zero frequency to the centre.
Fourier space       The image described as a sum of waves. "Frequency" counts
                    waves per Angstrom: high frequency means fine detail.
Gaussian simulator  The standard image model: projection, CTF and white
                    Gaussian noise only (SI S3).
Gumbel-max          A way to pick one of several options with given
                    probabilities: add random noise -log(-log u), with u
                    uniform, to the log-probabilities and take the largest.
Gumbel-softmax      The straight-through trick for "pick one of many", built on
                    Gumbel-max.
hyperparameter      A setting of the prior (an entry of phi).
Kumaraswamy         A distribution on [0, 1] with two shape numbers a and b,
                    like the Beta distribution but easy to draw from:
                    x = (1 - (1 - u)^(1/b))^(1/a), u uniform. a = b = 1 gives
                    the uniform distribution.
micrograph          One whole cryo-EM photograph. Particle images are small
                    boxes cut out of micrographs ("extraction").
pixel size          How many Angstrom one pixel covers.
power spectrum      How much image intensity sits at each spatial scale.
                    Also called the PSD.
prior               The rules for drawing random settings for each image.
quaternion          Four numbers describing a 3-D rotation (the viewing direction).
radius of gyration  A measure of size: the root-mean-square distance of the
                    points from their centre.
reparameterised     A random draw written as "fixed random number pushed through
                    a formula of the settings", so that derivatives with respect
                    to the settings exist.
representative      A bank can hold several structures for each conformer (its
                    axis R, for example several models of the same state);
                    each image picks one at random. Usually R = 1.
residue             One building block of the chain: an amino acid in a protein,
                    a nucleotide in a nucleic acid. The simulator keeps one point per
                    residue.
seed                A number that makes the random draws repeatable.
sigmoid             The S-shaped curve 1 / (1 + exp(-x)): it rises smoothly
                    from 0 to 1.
SNR                 Signal-to-noise ratio: signal variance divided by noise
                    variance. This file mostly stores log10(SNR); -2 means 1/100.
standardise         Shift and scale an image to mean 0 and standard deviation 1.
straight-through    A trick for yes/no or pick-one-of-many draws: the image uses
                    the hard choice, while the derivative is taken from a smooth
                    version of it, so derivatives reach the probabilities.
v3                  The paper's name for this simulator.


===============================================================================
MAP OF THIS FILE
===============================================================================
Part 1   Optional speed-up (torch.compile)
Part 2   Rotations
Part 3   Stage 1: projection of the molecule
Part 4   Stage 3: the microscope blur (CTF), with zero-padding
Part 5   Noise building blocks, noise spectrum measurement, signal-to-noise
Part 6   Standardisation
Part 7   The layout of theta (156) and phi (44)
Part 8   Loading 3-D structures (the conformer bank)
Part 9   Stage 5: the edge band
Part 10  Stage 2: clutter
Part 11  Stage 4: the noise mixture
Part 12  The sampler: phi -> theta
Part 13  The renderer: theta -> image
Part 14  The default prior (the paper's Table S3)


===============================================================================
KNOWN SIMPLIFICATIONS AND CHOICES TO BE AWARE OF
===============================================================================
- The electron wavelength is fixed at 0.019866 A.
- The zero-padding around the CTF is always sized for a defocus of 3.5
  micrometres, even when the defocus prior goes higher (Part 4).
- The coloured noise uses the full 2-D average power spectrum of the particles,
  not its radial average (Part 5).
- Each image's random point clouds for clutter are fixed by its own seed. But
  the noise fields, and the random radii of shell-shaped clouds, come from one
  random stream per batch, seeded by the first image's seed. In a different
  batch, the same theta row therefore gets a new noise draw (at the paper's
  SNRs, a different-looking image with the same settings) and slightly
  different shell-shaped clutter (see render).
- The per-dataset noise spectrum is read from a file. If that file is missing,
  the code quietly falls back to an average spectrum, then to a simple formula
  (see Part 11).

Reproducibility: leave the environment variable CRYOSBI_COMPILE unset (the
default), so that torch.compile is not used and results do not depend on a
compiler. The last digits of an image can still differ between PyTorch versions,
and between CPU and GPU.
"""
from __future__ import annotations

import os
import random
from collections import namedtuple
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ==============================================================================================
# PART 1 -- OPTIONAL SPEED-UP
# ==============================================================================================
# PyTorch can "compile" a few small, often-called functions into faster code. This is OFF by
# default, so results do not depend on whether a compiler is available. To switch it on, set the
# environment variable CRYOSBI_COMPILE=1 before starting Python. Compiled code is faster, but its
# last digits can differ slightly from the uncompiled code. (Even uncompiled, the last digits can
# differ between PyTorch versions, and between CPU and GPU.)

_DISABLED = os.environ.get("CRYOSBI_COMPILE", "") in ("", "0")
_configured = False


def _configure() -> None:
    """Prepare torch.compile once: allow it to recompile when input sizes change.

    Without this, a change of data type or size could make PyTorch silently stop
    compiling. It runs only the first time a compiled function is called.
    """
    global _configured
    if _configured:
        return
    import torch._dynamo

    torch._dynamo.config.recompile_limit = max(
        getattr(torch._dynamo.config, "recompile_limit", 8), 64
    )
    torch._dynamo.config.accumulated_recompile_limit = max(
        getattr(torch._dynamo.config, "accumulated_recompile_limit", 256), 512
    )
    _configured = True


def maybe_compile(fn):
    """Wrap `fn` so it is compiled when compiling is switched on, and run normally otherwise.

    If compiling fails for any reason (for example no C++ compiler on the machine), the
    wrapper quietly falls back to the normal, uncompiled function and keeps using it.
    The plain function stays available as `wrapper.eager`.
    """
    state: dict = {"compiled": None, "failed": False}

    def wrapper(*args):
        if _DISABLED or state["failed"]:
            return fn(*args)
        try:
            if state["compiled"] is None:
                _configure()
                state["compiled"] = torch.compile(fn, dynamic=True)
            return state["compiled"](*args)
        except Exception:
            state["failed"] = True
            return fn(*args)

    wrapper.eager = fn
    return wrapper


# ==============================================================================================
# PART 2 -- ROTATIONS
# ==============================================================================================
# A rotation in 3-D is stored as a "unit quaternion": four numbers (w, x, y, z) whose squares
# add up to 1. Drawing the four numbers from a normal distribution and dividing by their length
# gives rotations that are spread evenly over all directions, which is what we want for
# particles that land on the grid in random orientations.

def quaternions_to_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """Turn unit quaternions (B, 4), in (w, x, y, z) order, into 3x3 rotation matrices (B, 3, 3)."""
    rotation_matrices = torch.zeros(
        (quaternions.shape[0], 3, 3),
        device=quaternions.device,
        dtype=quaternions.dtype,
    )

    rotation_matrices[:, 0, 0] = 1 - 2 * (quaternions[:, 2] ** 2 + quaternions[:, 3] ** 2)
    rotation_matrices[:, 0, 1] = 2 * (quaternions[:, 1] * quaternions[:, 2] - quaternions[:, 3] * quaternions[:, 0])
    rotation_matrices[:, 0, 2] = 2 * (quaternions[:, 1] * quaternions[:, 3] + quaternions[:, 2] * quaternions[:, 0])

    rotation_matrices[:, 1, 0] = 2 * (quaternions[:, 1] * quaternions[:, 2] + quaternions[:, 3] * quaternions[:, 0])
    rotation_matrices[:, 1, 1] = 1 - 2 * (quaternions[:, 1] ** 2 + quaternions[:, 3] ** 2)
    rotation_matrices[:, 1, 2] = 2 * (quaternions[:, 2] * quaternions[:, 3] - quaternions[:, 1] * quaternions[:, 0])

    rotation_matrices[:, 2, 0] = 2 * (quaternions[:, 1] * quaternions[:, 3] - quaternions[:, 2] * quaternions[:, 0])
    rotation_matrices[:, 2, 1] = 2 * (quaternions[:, 2] * quaternions[:, 3] + quaternions[:, 1] * quaternions[:, 0])
    rotation_matrices[:, 2, 2] = 1 - 2 * (quaternions[:, 1] ** 2 + quaternions[:, 2] ** 2)

    return rotation_matrices


# ==============================================================================================
# PART 3 -- STAGE 1: PROJECTION OF THE MOLECULE  (paper: Eq. S14)
# ==============================================================================================
# The molecule is a list of points: one per amino acid (its C-alpha atom) or one per nucleotide
# (its phosphorus atom). Each point becomes a small round Gaussian blob of width sigma (in
# Angstrom). We rotate the points, drop the depth coordinate (we look straight down the z axis),
# and add up the blobs on a square grid of N x N pixels. The result is a "projection": what the
# molecule would look like with a perfect microscope and no noise.
#
# Axes: in an image img[i, j], the first index i runs along x and the second index j along y.
# Shown with matplotlib's imshow, x therefore points down and y to the right. The "top" and
# "left" edges of the edge band (Part 9) are row 0 and column 0 of that picture.

def _optics_key(t):
    """A dictionary key for a number or a tensor, used by the small caches below."""
    return ("t", t.data_ptr(), t.device.type, t.device.index) if torch.is_tensor(t) else ("v", t)


# The pixel coordinates depend only on the box size and the pixel size, so they are computed once
# and kept here for reuse.
_GRID_CACHE: dict = {}


def _pixel_grid(num_pixels, pixel_size, device, dtype) -> torch.Tensor:
    """Coordinates, in Angstrom, of the N pixels along one side of the box, shape (1, N).

    Pixel p sits at (p - N/2) * pixel_size, so the box centre (pixel N/2) is at 0.
    """
    key = (_optics_key(num_pixels), _optics_key(pixel_size), str(device), dtype)
    entry = _GRID_CACHE.get(key)
    if entry is None:
        grid_min = -pixel_size * num_pixels * 0.5
        grid_max = pixel_size * num_pixels * 0.5
        n = int(num_pixels)
        grid = torch.arange(grid_min, grid_max, pixel_size, device=device)[0:n].reshape(1, n)
        entry = (grid.to(dtype), num_pixels, pixel_size)   # keep the inputs alive so the key stays valid
        _GRID_CACHE[key] = entry
    return entry[0]


def gen_rot_matrix(quats: torch.Tensor) -> torch.Tensor:
    """Same as quaternions_to_rotation_matrices (kept under this name for older code)."""
    return quaternions_to_rotation_matrices(quats)


def _gauss_axes(
    grid: torch.Tensor,      # (1, N) or (B, N): pixel coordinates
    coord_x: torch.Tensor,   # (B, A): x position of every atom
    coord_y: torch.Tensor,   # (B, A): y position of every atom
    neg_half_inv_var: torch.Tensor,   # (B, 1, 1): equals -1 / (2 sigma^2)
    mask: torch.Tensor,      # (B, A): 1 for a real atom, 0 for padding
):
    """The two 1-D Gaussian profiles of every atom, along x and along y.

    A round 2-D Gaussian is a product of an x-profile and a y-profile, so the whole image can
    be built from these two smaller tables. This is much faster than drawing every blob
    pixel by pixel. Padding atoms (mask 0) get an x-profile of exactly zero.
    """
    dx = grid.unsqueeze(-1) - coord_x.unsqueeze(1)
    dy = grid.unsqueeze(-1) - coord_y.unsqueeze(1)
    gauss_x = torch.exp(dx * dx * neg_half_inv_var) * mask.unsqueeze(1)
    gauss_y = torch.exp(dy * dy * neg_half_inv_var)
    return gauss_x, gauss_y


_gauss_axes_fast = maybe_compile(_gauss_axes)


def project_density(
    coords: torch.Tensor,
    quats: torch.Tensor,
    sigma: torch.Tensor,
    shift: torch.Tensor,
    num_pixels: int,
    pixel_size: float,
) -> torch.Tensor:
    """Stage 1: rotate each molecule, then sum one Gaussian blob per atom into an N x N image.

    Inputs
        coords      (B, 3, A) atom coordinates in Angstrom, centred on the molecule. Molecules
                    with fewer atoms are padded with non-finite values (inf or NaN), which are
                    ignored.
        quats       (B, 4) unit quaternions: the viewing direction of each image.
        sigma       (B, 1, 1) blob width in Angstrom.
        shift       (B, 2) in-plane shift in Angstrom (0 in the paper).
        num_pixels  N, the box size in pixels.
        pixel_size  Angstrom per pixel.
    Output
        (B, N, N) projection images. Each image sums to about 1 / pixel_size^2, because the
        blobs together carry a total weight of 1, split equally over the atoms
        (Eq. S14: the 1 / (2 pi sigma^2 A) factor).
    """

    num_batch, _, _ = coords.shape
    valid = torch.isfinite(coords).all(dim=1)
    mask = valid.to(coords.dtype)
    num_valid_atoms = mask.sum(dim=1, keepdim=True)
    norm = 1 / (2 * torch.pi * sigma.squeeze() ** 2 * num_valid_atoms.squeeze())

    # Replace padding atoms by a harmless 0 BEFORE any arithmetic. Cleaning up afterwards would be
    # enough for the image, but not for the derivatives: a single non-finite padding atom would
    # turn the derivative with respect to sigma into NaN. The mask then removes these atoms exactly.
    coords = torch.where(valid.unsqueeze(1), coords, coords.new_zeros(()))

    rot_matrix = gen_rot_matrix(quats)
    grid = _pixel_grid(num_pixels, pixel_size, coords.device, coords.dtype)

    coords_rot = torch.bmm(rot_matrix, coords)
    coords_rot[:, :2, :] += shift.unsqueeze(-1)

    # image[p, q] = sum over atoms of gauss_x[p, atom] * gauss_y[q, atom]: one matrix product.
    # A padding atom has gauss_x = 0, so it adds exactly nothing.
    gauss_x, gauss_y = _gauss_axes_fast(
        grid, coords_rot[:, 0, :], coords_rot[:, 1, :], (-0.5) / (sigma * sigma), mask
    )
    return torch.bmm(gauss_x, gauss_y.transpose(1, 2)) * norm.reshape(-1, 1, 1)


# ==============================================================================================
# PART 4 -- STAGE 3: THE MICROSCOPE BLUR (CTF)  (paper: Eq. S16)
# ==============================================================================================
# A real microscope does not record the projection directly. It records the projection after a
# "contrast transfer function" (CTF): in Fourier space, every spatial frequency f is multiplied by
#
#     C(f) = -(1/a) * [ a * cos(chi) + sqrt(1 - a^2) * sin(chi) ] * exp(-B |f|^2 / 2),
#     chi  = pi * lambda * defocus * |f|^2,
#
# with a the amplitude contrast (0.1), lambda the electron wavelength (0.019866 A, see the
# simplifications listed at the top), "defocus" in Angstrom, and B the blur strength (B-factor).
# The cos/sin part makes the familiar rings ("Thon rings"); the exp(...) part fades fine detail.
# Dividing by a only rescales the whole image, which the later steps undo.
#
# Multiplying in Fourier space wraps signal around the edges of the box: blur that leaves the
# right edge would come back in on the left. Experimental particle images do not do that, so the image
# is first surrounded by zeros ("zero-padding"), blurred, and then cut back to N x N.

_FFT_FRIENDLY_FACTORS = (2, 3, 5, 7)


# |f|^2 on the half-plane used by the real Fourier transform, kept for reuse. It depends only on
# the padded box size and the pixel size.
_FREQ2_CACHE: dict = {}


def _freq2_half_plane(n_fft: int, pixel_size, device, dtype) -> torch.Tensor:
    """Squared spatial frequency |f|^2 (in 1/A^2) on the half-plane of `rfft2`, shape (n_fft, n_fft // 2 + 1)."""
    key = (int(n_fft), float(pixel_size), str(device), dtype)
    cached = _FREQ2_CACHE.get(key)
    if cached is None:
        psz = float(pixel_size)
        fy = torch.fft.fftfreq(n_fft, d=psz, device=device, dtype=dtype)    # full, along dim -2
        fx = torch.fft.rfftfreq(n_fft, d=psz, device=device, dtype=dtype)   # half, along dim -1
        cached = fy.unsqueeze(-1) ** 2 + fx.unsqueeze(0) ** 2
        _FREQ2_CACHE[key] = cached
    return cached


def next_fast_len(n: int) -> int:
    """The smallest whole number >= n whose only prime factors are 2, 3, 5 and 7.

    Fourier transforms are much faster for such sizes, so the padded box is rounded up to one.
    """
    n = int(max(n, 1))
    while True:
        m = n
        for f in _FFT_FRIENDLY_FACTORS:
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def ctf_pad_px(
    num_pixels: int,
    pixel_size: float = 1.0,
    defocus_max_um: float = 3.5,
    wavelength_A: float = 0.019866,
    frac_min: float = 0.5,
    frac_max: float = 0.75,
    k_eff_frac: float = 0.35,
) -> int:
    """How many zero pixels to add on EACH side of the box before applying the CTF.

    Defocus spreads ("delocalises") signal sideways. The spread is largest at the highest
    frequency the pixels can hold, and in pixels it is about

        d_px = defocus_max * wavelength / (2 * pixel_size^2).

    We pad by 0.35 * d_px (the B-factor fades the very highest frequencies, so a fraction is
    enough; 0.35 was chosen by comparing against a much larger padding), keep the pad between
    N/2 and 3N/4, and round the padded box up to a size with fast Fourier transforms.

    `defocus_max_um` is 3.5 micrometres by default, and `render` always uses that default,
    whatever the defocus prior is. For images with a larger defocus (the priors of Table S3 reach
    5.0 and 5.62 micrometres), the padding can therefore be smaller than this rule would give.
    The result depends only on N and the pixel size.
    """
    n = int(num_pixels)
    psz = float(pixel_size) if float(pixel_size) > 0 else 1.0
    d_px = float(defocus_max_um) * 1e4 * float(wavelength_A) / (2.0 * psz * psz)
    pad = k_eff_frac * d_px
    pad = min(max(pad, frac_min * n), frac_max * n)
    target = next_fast_len(int(round(n + 2.0 * pad)))
    return max((target - n) // 2, 0)


def _ctf_tensor(freq2_2d, defocus, b_factor, amp):
    """The CTF value C(f) at every frequency of the half-plane (Eq. S16).

    `defocus` is in micrometres (times 10000 gives Angstrom), `b_factor` in A^2, `amp` is the
    amplitude contrast a. The order of the operations is kept on purpose: rewriting the formula
    (for example cancelling the leading `amp` against the final `/ amp`) gives the same result on
    paper but different rounding on a computer.
    """
    phase = defocus * torch.pi * 2.0 * 10000 * 0.019866   # 0.019866 A: the electron wavelength used throughout
    theta = phase * freq2_2d * 0.5                        # chi = pi * lambda * defocus * |f|^2
    env = torch.exp(-b_factor * freq2_2d * 0.5)           # exp(-B |f|^2 / 2)
    return (
        -amp * torch.cos(theta) - torch.sqrt(1 - amp**2) * torch.sin(theta)
    ) * env / amp


_ctf_tensor_fast = maybe_compile(_ctf_tensor)


def apply_ctf(
    image: torch.Tensor, defocus, b_factor, amp, pixel_size, pad_px: int = 0
) -> torch.Tensor:
    """Stage 3: blur images with the microscope's CTF.

    Inputs
        image       (B, N, N) images, before noise.
        defocus     (B, 1, 1) defocus in micrometres.
        b_factor    (B, 1, 1) B-factor in A^2.
        amp         (B, 1, 1) amplitude contrast (0.1 in the paper).
        pixel_size  Angstrom per pixel.
        pad_px      zeros added on each side before the Fourier transform and cut away after
                    (see ctf_pad_px). 0 means no padding, which lets blurred signal wrap
                    around the box edges.
    Output
        (B, N, N) blurred images.

    Zeros are the natural padding here because the CTF is applied to the clean signal, before
    any noise is added, and the simulated specimen has nothing outside the box. (A clutter
    object that crosses the box edge is cut off there: its part outside the box is not
    simulated.)
    """

    num_batch, num_pixels, _ = image.shape
    p = int(pad_px)
    if p > 0:
        image = torch.nn.functional.pad(image, (p, p, p, p))
    n_fft = image.shape[-1]

    # The image is real and the CTF is symmetric, so the "real" Fourier transform (rfft2), which
    # stores only half of the frequencies, gives the same result at about half the cost.
    freq2_2d = _freq2_half_plane(n_fft, pixel_size, image.device, image.dtype)

    ctf = _ctf_tensor_fast(freq2_2d, defocus, b_factor, amp)

    conv_image_ctf = torch.fft.rfft2(image) * ctf
    image_ctf = torch.fft.irfft2(conv_image_ctf, s=(n_fft, n_fft))

    if p > 0:
        image_ctf = image_ctf[..., p : p + num_pixels, p : p + num_pixels]
    return image_ctf


# ==============================================================================================
# PART 5 -- NOISE BUILDING BLOCKS, NOISE-SPECTRUM MEASUREMENT, SIGNAL-TO-NOISE
# ==============================================================================================
# Three kinds of noise field are mixed in Stage 4 (Part 11). This part makes each one, scaled to
# variance 1:
#   - white Gaussian noise: the same strength at every spatial scale;
#   - coloured noise: white noise reshaped to the power spectrum measured on experimental particles
#     (paper Eq. S11);
#   - Poisson "shot" noise: noise from counting electrons, stronger where the image is brighter
#     (paper Eq. S10).
# It also measures the power spectrum of experimental particle images (stored in MRC files), and turns a
# requested signal-to-noise ratio into a noise level (paper Eq. S15).

def power_spectrum(image: torch.Tensor):
    """Power spectrum |F(image)|^2 of each image, after subtracting the image's mean value.

    Input (..., N, N); output (..., N, N) in the usual (unshifted) FFT order.
    """
    image = image - image.mean(dim=(-2, -1), keepdim=True)
    freq = torch.fft.fft2(image, dim=(-2, -1))
    return torch.abs(freq) ** 2


# --- reading particle images from MRC files ----------------------------------------------------
# MRC is the standard file format for cryo-EM images. A ".mrcs" file usually holds a stack of
# particle images; a ".mrc" file may hold one image or a stack. These four helpers find the files,
# list every image in them, and load a chosen set. They need the `mrcfile` package.

# One image inside one file: (path of the file, position of the image in the file's stack).
MRCImageReference = namedtuple("MRCImageReference", "file_path inner_index")


def find_mrc_paths(root_folder: str | Path) -> list[Path]:
    """All .mrc and .mrcs files under `root_folder` (sub-folders included), in sorted order."""
    root = Path(root_folder)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".mrc", ".mrcs"))


def build_mrc_image_index(paths: list[Path]) -> list[MRCImageReference]:
    """One MRCImageReference for every image in every file, in file order and stack order."""
    import mrcfile

    refs: list[MRCImageReference] = []
    for path in paths:
        with mrcfile.mmap(str(path), mode="r", permissive=True) as mrc:
            count = 1 if mrc.data.ndim == 2 else int(mrc.data.shape[0])
        refs.extend(MRCImageReference(file_path=str(path), inner_index=i) for i in range(count))
    return refs


def load_mrc_image_refs(refs: list[MRCImageReference]) -> torch.Tensor:
    """Load the listed images as one float32 tensor (B, N, N). All images must have the same size."""
    import mrcfile
    import numpy as np

    images = []
    open_files: dict = {}
    try:
        for ref in refs:
            mrc = open_files.get(ref.file_path)
            if mrc is None:
                mrc = mrcfile.mmap(ref.file_path, mode="r", permissive=True)
                open_files[ref.file_path] = mrc
            data = mrc.data
            plane = data if data.ndim == 2 else data[int(ref.inner_index)]
            images.append(torch.from_numpy(np.array(plane, dtype=np.float32)))
    finally:
        for mrc in open_files.values():
            mrc.close()
    return torch.stack(images)


def load_image_batch(image_refs: list[tuple[str, int]]) -> torch.Tensor:
    """Load images given as (file path, position in stack) pairs. Returns (B, N, N)."""
    refs = [MRCImageReference(file_path=file_path, inner_index=inner_idx) for file_path, inner_idx in image_refs]
    return load_mrc_image_refs(refs)


def find_mrc_files(root_folder: str | Path) -> list[Path]:
    """Same as find_mrc_paths."""
    return find_mrc_paths(root_folder)


def select_image_index(
    image_index: list[tuple[str, int]],
    num_images: int | None = 10000,
    selection_mode: str = "random",
    random_seed: int = 42,
) -> list[tuple[str, int]]:
    """Choose which images to use for measuring the spectrum.

    "random" takes `num_images` images at random (repeatably, through `random_seed`), "sequential"
    takes the first `num_images`, and num_images=None takes all of them.
    """
    if num_images is None:
        return image_index

    num_images = min(int(num_images), len(image_index))
    if selection_mode == "sequential":
        return image_index[:num_images]
    if selection_mode == "random":
        rng = random.Random(int(random_seed))
        return sorted(rng.sample(image_index, num_images))
    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def radial_power_spectrum(psd: torch.Tensor) -> torch.Tensor:
    """Average a 2-D power spectrum over rings of equal |f| (a "radial average").

    Input (..., N, N) in FFT order; output (..., R), one value per whole-pixel ring radius.
    """
    shifted = torch.fft.fftshift(psd, dim=(-2, -1))
    h, w = shifted.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=shifted.device),
        torch.arange(w, device=shifted.device),
        indexing="ij",
    )
    cy, cx = h // 2, w // 2
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).long()
    max_r = int(rr.max().item()) + 1

    rr_flat = rr.reshape(-1)
    flat = shifted.reshape(-1, h * w)
    counts = torch.bincount(rr_flat, minlength=max_r).clamp_min(1).to(flat.dtype)

    profiles = torch.zeros(flat.shape[0], max_r, device=flat.device, dtype=flat.dtype)
    for idx, spectrum in enumerate(flat):
        profiles[idx].scatter_add_(0, rr_flat, spectrum)
    profiles = profiles / counts.unsqueeze(0)
    return profiles.reshape(*psd.shape[:-2], max_r)


def estimate_power_law_alpha(
    reference_images: torch.Tensor,
    fit_min_radius: int = 4,
    fit_max_radius: int | None = 60,
):
    """Fit spectrum ~ 1 / radius^alpha to the images and return alpha (a straight-line fit in log-log)."""
    radial_psd = radial_power_spectrum(power_spectrum(reference_images)).mean(dim=0)
    radii = torch.arange(radial_psd.shape[0], device=radial_psd.device, dtype=torch.float32)
    if fit_max_radius is None:
        fit_max_radius = radial_psd.shape[0] - 1
    mask = (
        (radii >= fit_min_radius)
        & (radii <= fit_max_radius)
        & torch.isfinite(radial_psd)
        & (radial_psd > 0)
    )
    x = torch.log(radii[mask])
    y = torch.log(radial_psd[mask])
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum().clamp_min(1e-8)
    return float((-slope).item())


def estimate_empirical_noise_psd(
    root_folder: str | Path,
    num_images: int | None = 10000,
    selection_mode: str = "random",
    random_seed: int = 42,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Measure the average 2-D power spectrum of experimental particle images (paper Eq. S11).

    Reads the MRC files under `root_folder`, picks `num_images` particles (10,000 by default, at
    random with seed 42), and averages their power spectra. Returns an (N, N) tensor in FFT
    order. This 2-D average, not its radial average, is what colours the noise.
    """
    full_index = build_image_index(root_folder)
    if len(full_index) == 0:
        raise ValueError(f"No .mrc or .mrcs images found under {root_folder}")

    selected_index = select_image_index(
        full_index,
        num_images=num_images,
        selection_mode=selection_mode,
        random_seed=random_seed,
    )

    mean_psd = None
    total_images = 0
    device = torch.device(device)

    for start in range(0, len(selected_index), batch_size):
        image_refs = selected_index[start : start + batch_size]
        images = load_image_batch(image_refs).to(device)
        batch_psd = power_spectrum(images).mean(dim=0)
        if mean_psd is None:
            mean_psd = batch_psd
            total_images = len(image_refs)
        else:
            new_total = total_images + len(image_refs)
            mean_psd = (mean_psd * total_images + batch_psd * len(image_refs)) / new_total
            total_images = new_total

    return mean_psd


def build_image_index(root_folder: str | Path) -> list[tuple[str, int]]:
    """Every image under `root_folder`, as (file path, position in stack) pairs."""
    image_index = build_mrc_image_index(find_mrc_files(root_folder))
    return [(image_ref.file_path, image_ref.inner_index) for image_ref in image_index]


def estimate_empirical_noise_parameters(
    root_folder: str | Path,
    num_images: int | None = 10000,
    selection_mode: str = "random",
    random_seed: int = 42,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict:
    """Measure the noise spectrum of a data set and return everything a spectrum file stores.

    The returned dictionary can be saved with torch.save(...) and later given to the renderer
    as `empirical_cache_file` (see RenderShape.noise). Keys:
        mean_psd         (N, N) average power spectrum, the one that is used;
        radial_psd       its radial average, for plots;
        estimated_alpha  the slope of a power-law fit, for information;
        num_images_used  how many particles went into the average;
        selection_mode   how they were picked ("random" or "sequential");
        root_folder      the folder they were read from.
    The pixel size is not read from the files: give the particles' pixel size to RenderShape.
    """
    mean_psd = estimate_empirical_noise_psd(
        root_folder=root_folder,
        num_images=num_images,
        selection_mode=selection_mode,
        random_seed=random_seed,
        batch_size=batch_size,
        device=device,
    )
    radial_psd = radial_power_spectrum(mean_psd.unsqueeze(0)).squeeze(0)
    estimated_alpha = estimate_power_law_alpha(mean_psd.unsqueeze(0))
    return {
        "mean_psd": mean_psd,
        "radial_psd": radial_psd,
        "estimated_alpha": estimated_alpha,
        "num_images_used": len(
            select_image_index(
                build_image_index(root_folder),
                num_images=num_images,
                selection_mode=selection_mode,
                random_seed=random_seed,
            )
        ),
        "selection_mode": selection_mode,
        "root_folder": str(root_folder),
    }


# sqrt(spectrum / mean spectrum), prepared once per spectrum and kept for reuse.
_PSD_CACHE: dict = {}


# Exponent of the simple 1 / r^p spectrum used ONLY when no measured spectrum is available
# (the last fallback, Part 11). 0.39 is the slope of the average spectrum over our data sets.
DEFAULT_SYNTHETIC_PSD_EXPONENT = 0.39


# Default dose scale of _poisson_noise_unit when it is called on its own. The renderer never uses
# this default: it always passes gamma = 43 (POISSON_GAMMA, Part 11).
DEFAULT_POISSON_DOSE_SCALE = 5.0


def circular_mask(n_pixels: int, radius: int, device: str = "cpu") -> torch.Tensor:
    """(N, N) True/False mask of the pixels closer than `radius` to the box centre."""
    grid = torch.linspace(
        -0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels, device=device
    )
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    return r_2d < radius**2


def get_snr(images, snr):
    """The noise level sigma_n that gives each image the requested signal-to-noise ratio (Eq. S15).

    `images` (B, N, N) are the clean images, after the CTF; `snr` is log10(SNR). The signal is
    measured as the standard deviation of the pixels inside the circle that touches the box edges
    (background pixels included). Then

        sigma_n = std(signal in the circle) / sqrt(10 ** log10_snr),

    so SNR = (signal variance) / (noise variance). Output shape (B, 1, 1).
    """
    mask = circular_mask(
        n_pixels=images.shape[-1],
        radius=images.shape[-1] // 2,
        device=images.device,
    )
    signal_power = torch.std(images[:, mask], dim=[-1])
    assert signal_power.shape[0] == images.shape[0]
    return signal_power.reshape(-1, 1, 1) / torch.sqrt(torch.pow(torch.tensor(10), snr))


# Measured spectra, kept in memory after the first read so files are read only once.
_EMPIRICAL_NOISE_CACHE = {}


def get_empirical_noise_parameters_cached(
    root_folder: str | Path,
    num_images: int | None = 10000,
    selection_mode: str = "random",
    random_seed: int = 42,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    cache_file: str | Path | None = None,
) -> dict:
    """Get a data set's noise spectrum: from `cache_file` if one is given, otherwise by measuring it.

    Either way the result is kept in memory, so later calls are free.
    """
    if cache_file is not None:
        cache_key = ("file", str(cache_file))
        if cache_key not in _EMPIRICAL_NOISE_CACHE:
            _EMPIRICAL_NOISE_CACHE[cache_key] = load_empirical_noise_parameters(
                cache_file,
                map_location="cpu",
            )
        return _EMPIRICAL_NOISE_CACHE[cache_key]

    cache_key = (
        str(root_folder),
        num_images,
        selection_mode,
        int(random_seed),
        int(batch_size),
        str(device),
    )
    if cache_key not in _EMPIRICAL_NOISE_CACHE:
        _EMPIRICAL_NOISE_CACHE[cache_key] = estimate_empirical_noise_parameters(
            root_folder=root_folder,
            num_images=num_images,
            selection_mode=selection_mode,
            random_seed=random_seed,
            batch_size=batch_size,
            device=device,
        )
    return _EMPIRICAL_NOISE_CACHE[cache_key]


def load_empirical_noise_parameters(
    input_file: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Read a spectrum file written with torch.save(estimate_empirical_noise_parameters(...))."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Empirical noise cache file not found: {input_file}")
    stats = torch.load(input_file, map_location=map_location)
    required = {"mean_psd", "radial_psd", "estimated_alpha", "num_images_used"}
    missing = required.difference(stats.keys())
    if missing:
        raise ValueError(
            f"Empirical noise cache file is missing required keys: {sorted(missing)}"
        )
    return stats


def _prepared_psd(psd: torch.Tensor, device, ndim: int) -> torch.Tensor:
    """sqrt(psd / mean(psd)), on the right device and shaped to broadcast over a batch. Cached."""
    key = (psd.data_ptr(), psd.device.type, psd.device.index, psd.dtype,
           tuple(psd.shape), str(device), ndim)
    entry = _PSD_CACHE.get(key)
    if entry is None:
        prepared = psd.to(device)
        prepared = torch.sqrt(prepared / prepared.mean())
        prepared = prepared.reshape([1] * (ndim - 2) + list(psd.shape))
        entry = (prepared, psd)
        _PSD_CACHE[key] = entry
    return entry[0]


def colored_fft_noise_from_psd(image: torch.Tensor, psd: torch.Tensor, seed=None):
    """Coloured noise with the measured spectrum `psd`, scaled to mean 0 and variance 1 (Eq. S11).

    Draw white noise, take its Fourier transform, multiply every frequency by
    sqrt(psd / mean(psd)), transform back. Only the strength of each frequency changes; the
    random phases of the white noise are kept. Output has the shape of `image`.
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(image)
    psd_root = _prepared_psd(psd, image.device, image.ndim)
    noise_f = torch.fft.fft2(noise, dim=(-2, -1))
    noise = torch.real(torch.fft.ifft2(noise_f * psd_root, dim=(-2, -1)))
    noise = noise - noise.mean(dim=(-2, -1), keepdim=True)
    noise = noise / noise.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return noise


def _poisson_noise_unit(image: torch.Tensor, snr, dose_scale: float = DEFAULT_POISSON_DOSE_SCALE, seed=None):
    """Shot noise from counting electrons, scaled to mean 0 and variance 1 (Eq. S10).

    The clean image I (after the CTF, clutter included) is shifted and scaled to
    I~ = (I - min I) / max I. The expected electron count in a pixel is lambda * I~, with the dose
    scale lambda = clip(dose_scale * snr, 1, 200). We draw Poisson counts, subtract their expected
    value and divide by lambda, so brighter pixels get more noise.

    Note: I~ is not between 0 and 1. After the CTF, min I is negative, so I~ peaks at roughly 2-5
    for a particle and the background sits near |min I| / max I. The paper's Eq. S10 uses this
    same I~.

    `snr` here is whatever the caller passes; the renderer passes the LINEAR SNR, 10**log10_snr.
    """
    if seed is not None:
        torch.manual_seed(seed)

    snr_tensor = torch.as_tensor(snr, dtype=image.dtype, device=image.device)
    dose = torch.clamp(snr_tensor * dose_scale, 1.0, 200.0)
    dose = dose.reshape(-1, 1, 1)

    image_min = image.amin(dim=(-2, -1), keepdim=True)
    image_max = image.amax(dim=(-2, -1), keepdim=True)
    img = image - image_min
    img = img / (image_max + 1e-8)

    # A rate must be finite and >= 0. An empty image could give 0/0 = NaN above; this guard turns
    # such values into 0. For normal images it changes nothing.
    counts = (torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0) * dose).clamp_min(0.0)
    noisy_counts = torch.poisson(counts)
    noise = (noisy_counts - counts) / dose
    noise = noise - noise.mean(dim=(-2, -1), keepdim=True)
    noise = noise / noise.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return noise


def _gaussian_noise_unit(image: torch.Tensor, seed=None):
    """White Gaussian noise with mean 0 and variance 1, shaped like `image`."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn_like(image, device=image.device)


# ==============================================================================================
# PART 6 -- STANDARDISATION  (paper: SI S3, Stage 5)
# ==============================================================================================

def gaussian_normalize_image(images: torch.Tensor) -> torch.Tensor:
    """Shift and scale every image to mean 0 and standard deviation 1.

    Input and output (B, N, N). This is the last step of the simulator, so the overall
    brightness of an image carries no information; only its pattern does.
    """

    mean = images.mean(dim=[1, 2])
    std = images.std(dim=[1, 2])
    if (std == 0).any():
        raise ValueError("an image has zero standard deviation and cannot be standardised")
    return (images - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)


# ==============================================================================================
# PART 7 -- THE LAYOUT OF theta (156 NUMBERS) AND phi (44 NUMBERS)
# ==============================================================================================
# theta and phi are plain vectors of numbers. The two tables below say which position in the
# vector means what. Each row ("slot") has a name, a size (how many numbers) and a short
# description. theta's rows also carry three labels (dtype, transform, group) that are notes for
# the reader only: the code never reads them, and every entry of theta is stored as a
# floating-point number. phi's rows carry a `kind` (how the number is stored, used by
# pack_phi / unpack_phi) and a `group` (a note only). The paper lists the same quantities in
# Table S2 (theta, bottom block) and Table S3 (phi). Use slot_index("name") to find a slot's
# position, or unpack_theta / unpack_phi to get everything as named pieces.

# Fixed sizes the layout is built on.
N_SLOTS = 5           # at most 5 clutter objects per image (each object uses one "slot")


N_ARCHETYPES = 5      # five basic shapes for random clutter: compact, shell, filament, branched, sheet


N_SIDES = 4           # the edge band can sit on one of four sides: top, bottom, left, right


N_COUNT_BINS = 6      # the number of clutter objects is 0, 1, 2, 3, 4 or 5


# Codes stored in theta["noise_regime"]: which kind of noise mixture an image got (Part 12).
REGIME_FULL_PSD = 0   # only coloured noise
REGIME_WHITE = 1      # only white noise (the kind of noise of the Gaussian simulator)
REGIME_BRIDGE = 2     # a mix of all three


# Order of the three noise channels everywhere in this file, including theta["noise_w"].
NOISE_CHANNELS = ("fft", "poisson", "gaussian")   # coloured, shot noise, white


Slot = namedtuple("Slot", "name size dtype transform group doc")


# ---------------------------------------------------------------------------
# theta: the settings of ONE image. 156 numbers. (Paper: Table S2, bottom block.)
# ---------------------------------------------------------------------------
THETA_LAYOUT: tuple[Slot, ...] = (
    Slot("seed", 1, "int", "seed", "meta",
         "A whole number. It regenerates this image's fixed random point cloud for "
         "clutter (see derive_u). The noise fields and the shell radii come from one "
         "random stream per batch, seeded by the FIRST image's seed (see render)."),
    Slot("conformer_idx", 2, "int", "index", "base",
         "Which structure from the bank to show: (conformer number, representative "
         "number)."),
    Slot("quaternion", 4, "float", "unit_quaternion", "base",
         "The viewing direction, as a unit quaternion (w, x, y, z). Uniform over all "
         "rotations."),
    Slot("sigma", 1, "float", "identity", "base",
         "Width (standard deviation) of each Gaussian bead, in Angstrom. Used for the "
         "molecule and for its clutter alike."),
    Slot("shift", 2, "float", "identity", "base",
         "Sideways shift of the molecule in the box, in Angstrom. Always 0 in the "
         "paper, which is why it has no entry in phi."),
    Slot("defocus", 1, "float", "identity", "base", "Defocus in micrometres."),
    Slot("b_factor", 1, "float", "identity", "base",
         "B-factor of the CTF (how fast fine detail fades), in Angstrom^2."),
    Slot("amp", 1, "float", "identity", "base",
         "Amplitude contrast of the CTF. Fixed at 0.1, which is why it has no entry "
         "in phi."),
    Slot("log10_snr", 1, "float", "log10", "base",
         "log10 of the signal-to-noise ratio (-2 means SNR = 0.01). The noise level "
         "(get_snr) uses this number; the shot-noise dose (Eq. S10) uses the linear "
         "SNR, 10 ** log10_snr."),

    # --- noise ---------------------------------------------------------------
    Slot("noise_w", 3, "float", "identity", "noise",
         "The final weights of the three noise fields, in NOISE_CHANNELS order "
         "(coloured, shot noise, white), exactly as they are multiplied in Stage 4. "
         "The regime below has already been applied to them."),
    Slot("noise_regime", 1, "int", "categorical", "noise",
         "Which noise regime the image got: 0 = coloured only (1, 0, 0), 1 = white "
         "only (0, 0, 1), 2 = a mix. Kept only as a record; the weights above already "
         "contain it."),
    Slot("noise_white_flag", 1, "int", "gate", "noise",
         "1 when the coloured-noise weight came from its 'mostly white' branch (the "
         "bottom quarter of its range). Also only a record."),

    # --- edge band -----------------------------------------------------------
    Slot("band_present", 1, "int", "gate", "band",
         "1 when this image has an edge band, otherwise 0."),
    Slot("band_width_frac", 1, "float", "identity", "band",
         "How deep the band reaches into the image, as a fraction of the box size N. "
         "0 means no band."),
    Slot("band_side", 1, "int", "index", "band",
         "Which edge carries the band: 0 top, 1 bottom, 2 left, 3 right."),
    Slot("band_side_onehot", 4, "float", "onehot", "band",
         "The same side as four numbers (one of them 1). Written so that "
         "derivatives can reach the side probabilities in phi."),

    # --- clutter: one entry per slot, 5 slots ---------------------------------
    Slot("cl_present", N_SLOTS, "float", "gate", "clutter",
         "1 if this clutter slot holds an object, 0 if it is empty. With k objects, "
         "the first k slots are full."),
    Slot("cl_t_pdb", N_SLOTS, "float", "identity", "clutter",
         "The object's type t between 0 and 1: 0 = a random point cloud, 1 = a whole "
         "copy of the molecule, in between = a piece of the molecule holding a "
         "fraction t of its atoms (at least 7), blended with a point cloud."),
    Slot("cl_win_start", N_SLOTS, "float", "identity", "clutter",
         "A number in [0, 1) that picks the atom around which a piece of the molecule "
         "is cut."),
    Slot("cl_model_idx", N_SLOTS * 2, "int", "index", "clutter",
         "Which structure each copy or piece comes from. Always the image's own "
         "conformer."),
    Slot("cl_contrast_proc", N_SLOTS, "float", "identity", "clutter",
         "Brightness of the random point cloud. Can be negative: a hole in the ice, "
         "for example a gas bubble from radiation damage."),
    Slot("cl_amp_free", N_SLOTS, "float", "identity", "clutter",
         "A freely drawn brightness for a copy or piece, used when cl_use_free is 1."),
    Slot("cl_use_free", N_SLOTS, "float", "gate", "clutter",
         "1: the copy or piece uses cl_amp_free as its brightness. 0: its brightness "
         "is the fraction of atoms it holds. Either way it is then multiplied by t "
         "(Part 10), so with 0 a piece holding half the atoms adds a quarter of the "
         "molecule's density."),
    Slot("cl_topology", N_SLOTS * N_ARCHETYPES, "float", "simplex_raw", "clutter",
         "Five weights mixing the five basic shapes (compact, shell, filament, "
         "branched, sheet). They are divided by their sum before use."),
    Slot("cl_rg_A", N_SLOTS, "float", "identity", "clutter",
         "Size of the point cloud: its radius of gyration, in Angstrom."),
    Slot("cl_aniso", N_SLOTS * 2, "float", "log_scale", "clutter",
         "How much the point cloud is stretched along two of its axes (as a log), "
         "applied before it is rotated."),
    Slot("cl_hollowness", N_SLOTS, "float", "identity", "clutter",
         "Shell shape: 0 = a solid ball, 1 = a thin hollow shell."),
    Slot("cl_persistence", N_SLOTS, "float", "identity", "clutter",
         "Filament shape: how stiff the chain is (close to 1 = nearly straight)."),
    Slot("cl_d_fractal", N_SLOTS, "float", "identity", "clutter",
         "Branched shape: its fractal dimension (how quickly its mass grows with "
         "size)."),
    Slot("cl_sheet_thick", N_SLOTS, "float", "identity", "clutter",
         "Sheet shape: how thick the sheet is, relative to its width."),
    Slot("cl_quat", N_SLOTS * 4, "float", "unit_quaternion", "clutter",
         "Orientation of each clutter object, as a unit quaternion."),
    Slot("cl_offset_A", N_SLOTS * 2, "float", "identity", "clutter",
         "Position of each clutter object in the image plane, (x, y) in Angstrom "
         "from the box centre (x runs along the image's first index, see Part 3). "
         "It can lie outside the box."),
)


THETA_LEN = sum(s.size for s in THETA_LAYOUT)


# ---------------------------------------------------------------------------
# phi: the PRIOR, i.e. the rules for drawing theta. 44 numbers. (Paper: Table S3.)
#
# Inside phi, every number is stored in an "unconstrained" form: any 44 numbers make a valid
# prior, never an impossible value. unpack_phi turns the stored form
# into the real, physical value; pack_phi does the reverse. The `kind` column says how:
#   logit        a probability p, stored as log(p / (1 - p));
#   logit_floor  a probability that must stay above a floor f (see PROB_FLOORS);
#   softplus     a positive number s, stored as log(exp(s) - 1);
#   ordered      a range [lo, hi], stored as two numbers (a, log(hi - lo)), so hi > lo always
#                holds. a is lo itself, except for the ranges listed in RANGE_LOWER_BOUNDS,
#                whose lower end must stay above a bound b: there a = log(exp(lo - b) - 1).
#                The noise-weight ranges (MIX_RANGE_*) are also capped at 1;
#   simplex      probabilities that add up to 1, stored as logarithms.
# You normally never see the stored form: write physical values in a dictionary and call
# pack_phi, as in example.py.
# ---------------------------------------------------------------------------
PhiSlot = namedtuple("PhiSlot", "name size kind group doc")


PHI_LAYOUT: tuple[PhiSlot, ...] = (
    # ---- optics, 8 numbers: each a range [lo, hi]; every image draws uniformly inside ----
    PhiSlot("SIGMA", 2, "ordered", "optics",
            "Range of the bead width sigma, in Angstrom."),
    PhiSlot("DEFOCUS", 2, "ordered", "optics",
            "Range of the defocus, in micrometres."),
    PhiSlot("B_FACTOR", 2, "ordered", "optics",
            "Range of the CTF B-factor, in Angstrom^2."),
    PhiSlot("SNR", 2, "ordered", "optics",
            "Range of log10(SNR). Uniform in log10, so every factor of ten in SNR is "
            "equally likely."),

    # ---- noise, 10 numbers (Stage 4) ----
    PhiSlot("MIX_RANGE_gaussian", 2, "ordered", "noise",
            "Range of the white-noise weight."),
    PhiSlot("MIX_RANGE_fft", 2, "ordered", "noise",
            "Range of the coloured-noise weight. This weight is the dial between "
            "white and coloured noise."),
    PhiSlot("MIX_RANGE_poisson", 2, "ordered", "noise",
            "Range of the shot-noise weight."),
    PhiSlot("CONC_fft", 1, "softplus", "noise",
            "Pushes the coloured-noise weight towards the top of its range: "
            "w = lo + (hi - lo) * u ** (1 / CONC_fft), with u uniform in [0, 1]."),
    PhiSlot("WHITE_FRAC_fft", 1, "logit", "noise",
            "Probability that the coloured-noise weight is instead drawn from the "
            "bottom quarter of its range (images that are mostly white)."),
    PhiSlot("REGIME_full_psd", 1, "logit", "noise",
            "Probability that an image has coloured noise only."),
    PhiSlot("REGIME_white", 1, "logit_floor", "noise",
            "Probability that an image has white noise only, the kind of noise of the "
            "Gaussian simulator. This number never goes below 0.02 "
            "(WHITE_REGIME_FLOOR). The actual share of white-only images is "
            "min(REGIME_white, 1 - REGIME_full_psd), so it stays at least 2 % as long "
            "as REGIME_full_psd stays below 0.98. A white-only image with no clutter "
            "and no band is a Gaussian-simulator image, up to two small differences (the "
            "zero-padding around the CTF, and the rescaling of each image's noise to "
            "variance exactly 1)."),

    # ---- clutter, 17 numbers (Stage 2) ----
    PhiSlot("count_probs", N_COUNT_BINS, "simplex", "clutter",
            "Probabilities of placing 0, 1, 2, 3, 4 or 5 clutter objects around an "
            "image. Some of them land outside the box (see radial_exponent)."),
    PhiSlot("p_bank", 1, "logit", "clutter",
            "Probability that an object contains some of the molecule (t > 0). "
            "Otherwise (1 - p_bank) it is a pure random point cloud (t = 0)."),
    PhiSlot("p_duplicate", 1, "logit", "clutter",
            "Probability that an object is a whole copy of the molecule (t = 1)."),
    PhiSlot("p_free_amp", 1, "logit", "clutter",
            "Probability that a copy or piece gets a freely drawn brightness instead "
            "of one set by its number of atoms."),
    PhiSlot("rg_range_A", 2, "ordered", "clutter",
            "Range of the point clouds' size (radius of gyration), in Angstrom."),
    PhiSlot("contrast_range", 2, "ordered", "clutter",
            "Range of the point clouds' brightness. Negative values make holes."),
    PhiSlot("amp_free_range", 2, "ordered", "clutter",
            "Range of the freely drawn brightness of copies and pieces."),
    PhiSlot("aniso_log_halfwidth", 1, "softplus", "clutter",
            "How much point clouds can be stretched: each log-stretch is uniform in "
            "[-value, +value]."),
    PhiSlot("radial_exponent", 1, "softplus", "clutter",
            "Where objects sit (Eq. S7): at a random angle and a distance "
            "r = r_out * u ** (1 / (2 a)) from the box centre, with u uniform and r_out "
            "half the box diagonal. a = 1 spreads them evenly over the disc of radius "
            "r_out, which reaches past the box edges: about 36 % of the objects then "
            "sit outside the box and show only partly or not at all. a > 1 pushes them "
            "outwards (at a = 1.26, about 43 % sit outside)."),

    # ---- edge band, 9 numbers (Stage 5) ----
    PhiSlot("barcode_prob", 1, "logit", "band",
            "Probability that an image has an edge band ('barcode' is the code's "
            "name for the band)."),
    PhiSlot("barcode_width_frac", 2, "ordered", "band",
            "Range of the band depth, as a fraction of the box size N."),
    PhiSlot("side_weights", N_SIDES, "simplex", "band",
            "Probabilities of the band being on the top, bottom, left or right edge."),
    PhiSlot("depth_a", 1, "softplus", "band",
            "First shape number a of the Kumaraswamy distribution of the band depth "
            "(a = b = 1 means uniform)."),
    PhiSlot("depth_b", 1, "softplus", "band",
            "Second shape number b of the Kumaraswamy distribution of the band depth."),
)


PHI_LEN = sum(s.size for s in PHI_LAYOUT)


def slot_index(name: str, layout=None) -> slice:
    """Where the slot called `name` sits in a theta or phi vector, as a slice.

    Example: theta[:, slot_index("defocus")] is the defocus of every image. Searches theta's
    layout first, then phi's, unless `layout` says which one.
    """
    layouts = (THETA_LAYOUT, PHI_LAYOUT) if layout is None else (layout,)
    for lay in layouts:
        off = 0
        for s in lay:
            if s.name == name:
                return slice(off, off + s.size)
            off += s.size
    raise KeyError(name)


# ==============================================================================================
# PART 8 -- LOADING 3-D STRUCTURES (THE CONFORMER BANK)
# ==============================================================================================
# The simulator draws each image's molecule from a "bank": a tensor of shape (M, R, 3, A) holding
# M conformers (3-D shapes), R representatives of each (several structures per conformer; usually
# R = 1), 3 coordinates (x, y, z, in Angstrom), and A atoms. Only one bead per residue is kept:
# the C-alpha atom for proteins, the phosphorus atom for nucleic acids. Conformers with fewer atoms are
# padded with `inf`, which the projection ignores.

# The atom kept for each residue. Use "name P" for nucleic acids.
DEFAULT_ATOM_SELECTION = "name CA"


def load_bank(
    pdb_dir: str | Path,
    atom_selection: str = DEFAULT_ATOM_SELECTION,
    *,
    pattern: str = "*.pdb",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a bank from a folder of PDB files: every file becomes one conformer.

    Each structure is first moved so that the centre of mass of ALL its atoms (waters, ligands
    and hydrogens included, if the file has them) sits at 0. Then only the atoms chosen by
    `atom_selection` are kept (C-alpha by default; use "name P" for nucleic acids), so their own centre can
    be slightly off 0. Files are read in sorted order, so conformer numbers are stable. Returns
    (M, 1, 3, A) in Angstrom. Needs MDAnalysis.
    """
    import MDAnalysis as mda

    files = sorted(Path(pdb_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {pdb_dir}")

    clouds: list[torch.Tensor] = []
    for f in files:
        univ = mda.Universe(str(f))
        univ.atoms.translate(-univ.atoms.center_of_mass())
        sel = univ.select_atoms(atom_selection)
        if sel.n_atoms == 0:
            raise ValueError(f"{f}: selection {atom_selection!r} matched no atoms")
        clouds.append(torch.from_numpy(sel.positions.T.copy()).to(torch.float32))

    return pad_bank(clouds, device=device)


def pad_bank(clouds, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Stack a list of (3, A_i) coordinate arrays into one (M, 1, 3, A_max) bank, padding with inf."""
    a_max = max(int(c.shape[1]) for c in clouds)
    out = torch.full((len(clouds), 1, 3, a_max), float("inf"), dtype=torch.float32)
    for i, c in enumerate(clouds):
        out[i, 0, :, : c.shape[1]] = c.to(torch.float32)
    return out.to(device)


def load_bank_pt(path: str | Path, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load a bank saved with torch.save, as (M, R, 3, A). A (M, 3, A) file is accepted too."""
    models = torch.load(path, map_location="cpu", weights_only=False)
    return as_4d(models).to(device).to(torch.float32)


def as_4d(models: torch.Tensor) -> torch.Tensor:
    """Turn a (M, 3, A) bank into (M, 1, 3, A); a (M, R, 3, A) bank is returned unchanged."""
    if models.ndim == 3:
        return models.unsqueeze(1)
    if models.ndim == 4:
        return models
    raise ValueError(
        f"bank must be (M, 3, A) or (M, R, 3, A), got {tuple(models.shape)}"
    )


def bank_envelope_A(models: torch.Tensor) -> float:
    """The largest distance of any atom from the origin (0, 0, 0), over the whole bank, in Angstrom.

    load_bank puts the origin at each structure's centre of mass.
    """
    m = as_4d(models)
    fin = torch.isfinite(m).all(dim=2)
    r = torch.where(fin, m.pow(2).sum(dim=2).sqrt(), torch.zeros_like(fin, dtype=m.dtype))
    return float(r.max().item())


# ==============================================================================================
# PART 9 -- STAGE 5: THE EDGE BAND  (paper: Eq. S12)
# ==============================================================================================
# When a particle sits close to the border of the micrograph, its box sticks out of the picture.
# Extraction software fills the missing part by repeating the last row (or column) of real
# pixels. Experimental data sets therefore contain some images with a "band" of identical lines along one
# edge. The simulator reproduces this: it copies the last uncut line, the one right next to the
# band, across a band of depth D pixels on one side. The band's edge is a smooth ramp (a sigmoid
# with scale tau_b = 0.5 pixel; it falls from 90 % to 10 % over about 2 pixels), so that
# derivatives with respect to D exist; D = 0 changes nothing at all.

_SIDE_TOP, _SIDE_BOTTOM, _SIDE_LEFT, _SIDE_RIGHT = 0, 1, 2, 3


def _edge_distance_stack(n: int, device, dtype) -> torch.Tensor:
    """(4, n, n): every pixel's distance, in pixels, to the top, bottom, left and right edge."""
    y = torch.arange(n, device=device, dtype=dtype).view(n, 1).expand(n, n)
    x = torch.arange(n, device=device, dtype=dtype).view(1, n).expand(n, n)
    last = float(n - 1)
    return torch.stack([y, last - y, x, last - x], dim=0)


def _replication_mask(
    depth_px: torch.Tensor,          # (B,) band depth in pixels; 0 means no band
    side: torch.Tensor,              # (B,) 0 top, 1 bottom, 2 left, 3 right
    n: int,
    tau: float = 0.5,
) -> torch.Tensor:
    """(B, n, n) mask M_e of Eq. S12: near 1 deep inside the band, near 0 away from it.

    M = sigmoid((D - distance_to_edge - 1/2) / tau). M is exactly 1/2 at distance D - 1/2 from
    the edge: for a whole-number D, halfway between the band's last line (distance D - 1) and
    the copied line (distance D; the depth is rounded to find it). With tau = 0.5 pixel, M
    falls from 90 % to 10 % over about 2 pixels: for D = 5, the lines at distance
    3, 4, 5, 6 get M = 0.95, 0.73, 0.27, 0.05. (On the copied line itself the blend changes
    nothing, because the band there equals that line.) Depth 0 gives a mask of exactly 0.
    """
    depth_px = depth_px.reshape(-1)
    dist = _edge_distance_stack(n, depth_px.device, depth_px.dtype)[side.reshape(-1)]
    gate = (depth_px > 0).to(depth_px.dtype).view(-1, 1, 1)
    return gate * torch.sigmoid((depth_px.view(-1, 1, 1) - dist - 0.5) / tau)


def apply_edge_replication(
    image: torch.Tensor,             # (B, N, N) the FINAL image: after the CTF and the noise
    *,
    depth_px: torch.Tensor,          # (B,) band depth in pixels; 0 means no band
    side: torch.Tensor,              # (B,) 0 top, 1 bottom, 2 left, 3 right
    src_line: torch.Tensor | None = None,   # leave as None: the copied line is found automatically
    side_onehot: torch.Tensor | None = None,  # (B, 4) one-hot side, lets derivatives reach the side probabilities
    tau: float = 0.5,
) -> torch.Tensor:
    """Stage 5: copy the last uncut line of each image across a band along one edge (Eq. S12).

    For a band of depth D along the top, rows 0 .. D-1 are the "missing" part and row D is the
    first real row, so row D is copied into the band. Along the bottom it is row N-1-D, and the
    same with columns for the left and right sides. Because the band copies the image's own
    pixels (noise included), along the edge it has the image's own values; across the depth
    every line is the same, which gives the striped "barcode" look.

    The result is image * (1 - M) + band * M, with M the soft mask. Only images that have a
    band are touched.
    """
    b, n, _ = image.shape
    depth_px = depth_px.reshape(-1).to(image.dtype)
    side = side.reshape(-1).long()
    if src_line is not None:               # None = find the line from the depth and the side
        src_line = src_line.reshape(-1).long().clamp(0, n - 1)

    # Bands are rare, so work only on the images that have one.
    active = (depth_px > 0).nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return image

    img_a = image.index_select(0, active)
    depth_a = depth_px.index_select(0, active)
    side_a = side.index_select(0, active)
    k = active.numel()

    # The copied line is the last line that was not cut; which one depends on the side.
    d_idx = depth_a.round().long().clamp(0, n - 1)
    src_from_start = d_idx                                   # top / left
    src_from_end = (n - 1 - d_idx).clamp(0, n - 1)           # bottom / right
    from_end = ((side_a == _SIDE_BOTTOM) | (side_a == _SIDE_RIGHT))
    src_a = torch.where(from_end, src_from_end, src_from_start)
    if src_line is not None:                                 # an explicit line, only for testing
        src_a = src_line.index_select(0, active).clamp(0, n - 1)

    idx = torch.arange(k, device=image.device)
    rows = img_a[idx, src_a, :]                          # (k, N) a horizontal line
    cols = img_a[idx, :, src_a]                          # (k, N) a vertical line
    band_from_row = rows.unsqueeze(1).expand(k, n, n)    # repeated down the depth
    band_from_col = cols.unsqueeze(2).expand(k, n, n)    # repeated across the depth

    if side_onehot is None:
        mask = _replication_mask(depth_a, side_a, n, tau=tau).to(image.dtype)
        is_vertical_edge = (side_a >= _SIDE_LEFT).view(-1, 1, 1)
        band = torch.where(is_vertical_edge, band_from_col, band_from_row)
    else:
        # The side comes as a one-hot vector from a straight-through draw: exactly one entry is 1
        # in the image, but derivatives can still reach the four side probabilities. Building the
        # mask as a weighted sum over the four sides is what lets those derivatives exist.
        y = side_onehot.index_select(0, active).to(image.dtype)          # (k, 4)
        mask = 0.0
        for kk in range(4):
            side_k = torch.full_like(side_a, kk)
            mask = mask + y[:, kk].view(-1, 1, 1) * _replication_mask(
                depth_a, side_k, n, tau=tau).to(image.dtype)
        w_vert = (y[:, _SIDE_LEFT] + y[:, _SIDE_RIGHT]).view(-1, 1, 1)
        band = (1.0 - w_vert) * band_from_row + w_vert * band_from_col

    out_a = (1.0 - mask) * img_a + mask * band
    # put the changed images back (written so that derivatives flow to both inputs)
    return image.index_copy(0, active, out_a)


# ==============================================================================================
# PART 10 -- STAGE 2: CLUTTER  (paper: SI S2 "Multiparticle clutter", Eq. S7-S9)
# ==============================================================================================
# A particle box rarely holds the molecule alone. It can also hold neighbouring copies, broken
# pieces, detergent, ice contamination or holes. The simulator adds 0 to 5 extra objects per
# image. Each object is a cloud of points drawn exactly like the molecule (one Gaussian bead of
# width sigma per point) and added BEFORE the microscope blur, so it is blurred the same way.
#
# An object's type is one number t between 0 and 1 (Eq. S8 and around it):
#   t = 0          a random point cloud ("procedural" object) shaped by five basic shapes;
#   t = 1          a whole second copy of the molecule;
#   0 < t < 1      a compact piece of the molecule holding a fraction t of its atoms (at least
#                  7), BLENDED, at the same position and orientation, with a random point cloud:
#                  the piece gets weight t times its brightness, and the cloud weight (1 - t)
#                  times its contrast.
# "Brightness" means total density compared with the molecule: 1 is as much as the whole
# molecule. Unless a free brightness is drawn (cl_use_free), a piece's brightness is the fraction
# of atoms it holds, about t, so its weight is about t * t: a piece holding half the atoms adds a
# quarter of the molecule's density (each of its atoms is drawn at half the molecule's strength).
# Objects are placed at a random angle and a random distance from the box centre (Eq. S7). Some
# land outside the box and show only partly or not at all.

ARCH_COMPACT, ARCH_SHELL, ARCH_FILAMENT, ARCH_BRANCHED, ARCH_SHEET = range(5)


ARCH_NAMES = ("compact", "shell", "filament", "branched", "sheet")


# Fixed settings of the clutter model (not priors, so not in phi). RenderShape reads them.
_V3_DEFAULTS = {
    "n_slots": 5,                   # up to 5 objects (a record only: the code uses N_SLOTS)
    "n_points_proc": 1024,          # points in one random point cloud (P = 1024 in the paper)
    "min_fragment_atoms": 7,        # a piece of the molecule has at least 7 residues (~2 helix turns)
    "count_tau": 0.5,               # smoothness of the straight-through draw of the object count
    "gate_tau": 0.1,                # smoothness of the straight-through yes/no draws
}


def project_points(
    coords: torch.Tensor,        # (B, 3, P) points, already rotated and moved into place
    weights: torch.Tensor,       # (B, P) brightness of each point (can be negative)
    sigma: torch.Tensor,         # (B, 1, 1) tensor, or a 0-d tensor for all: bead width in Angstrom
    num_pixels: int,
    pixel_size: float,
) -> torch.Tensor:
    """Draw weighted point clouds as Gaussian beads on an N x N grid. Returns (B, N, N).

    Same idea as project_density (Part 3), with two differences: every point has its own
    weight (which lets one object be brighter than another, or darker than the ice), and the
    points are already rotated and placed. Non-finite points are ignored.
    """
    B = coords.shape[0]
    dev = coords.device
    gmin = -pixel_size * num_pixels * 0.5
    grid = torch.arange(num_pixels, device=dev, dtype=coords.dtype) * pixel_size + gmin
    grid = grid.unsqueeze(0).expand(B, -1)                              # (B, N)

    finite = torch.isfinite(coords).all(dim=1)                          # (B, P)
    safe = torch.where(finite.unsqueeze(1), coords, torch.zeros_like(coords))
    w = torch.where(finite, weights, torch.zeros_like(weights))

    s = sigma if sigma.ndim == 0 else sigma.reshape(B, 1, 1)
    gx = torch.exp(-0.5 * ((grid.unsqueeze(-1) - safe[:, 0, :].unsqueeze(1)) / s) ** 2)
    gy = torch.exp(-0.5 * ((grid.unsqueeze(-1) - safe[:, 1, :].unsqueeze(1)) / s) ** 2)
    return torch.bmm(gx * w.unsqueeze(1), gy.transpose(1, 2))


# --- the five basic shapes of a random point cloud (Eq. S9) ---------------------------------------
# Each shape turns the SAME fixed set of random normal points u (P points, 3 coordinates) into a
# point cloud. Mixing the five shapes point by point gives intermediate shapes. The mix is rescaled
# to the drawn size afterwards, but the five shapes do not start equally large: their radius of
# gyration is 1 for compact, 1 to 1.24 for shell (solid to hollow), and about 1.73 (the square
# root of 3) for filament, branched and sheet. With equal weights, the last three therefore
# shape the mix more.

def _arch_compact(u: torch.Tensor) -> torch.Tensor:
    """Compact blob: x = u / sqrt(3). u is (B, P, 3)."""
    return u / (3.0 ** 0.5)


def _arch_shell(u: torch.Tensor, hollowness: torch.Tensor) -> torch.Tensor:
    """Shell, from a solid ball (hollowness h = 0) to a thin hollow shell (h = 1).

    Each point keeps its direction u/|u| and gets a new distance from the centre:
    (1 - h) * rho^(1/3) + h * (0.85 + 0.15 * rho^(1/3)), with rho uniform in [0, 1]. This stands
    for things like detergent micelles, nanodisc rims or small vesicles.
    Note: rho is drawn here from PyTorch's global random stream (seeded once per batch by render),
    not from the fixed per-image points u.
    """
    d = torch.nn.functional.normalize(u, dim=-1)                        # direction on the sphere
    r_uniform = torch.rand_like(u[..., :1]) ** (1.0 / 3.0)              # radius of a solid ball
    h = hollowness.reshape(-1, 1, 1).clamp(0.0, 1.0)
    r = (1.0 - h) * r_uniform + h * (0.85 + 0.15 * r_uniform)           # pushed towards the surface
    return d * r * 1.29                                                 # about unit size for a solid ball


def _arch_filament(u: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
    """Filament: a random chain whose step directions change slowly ("worm-like chain").

    s_n = p * s_(n-1) + (1 - p) * u_n, and the points are the running sum of the unit steps
    s_n / |s_n|. The persistence p sets the stiffness. Stands for unfolded chains, nucleic acid,
    fibrils or worm-like detergent. The result is centred and scaled so that its coordinates have
    standard deviation 1 (radius of gyration about 1.73).
    """
    B, P, _ = u.shape
    p = persistence.reshape(B, 1, 1).clamp(0.01, 0.999)
    # The steps depend on each other, so they are built one after the other in a loop.
    prev = u[:, 0]
    acc = [prev]
    for i in range(1, P):
        prev = p[:, 0] * prev + (1.0 - p[:, 0]) * u[:, i]
        acc.append(prev)
    steps = torch.stack(acc, dim=1)
    pts = torch.cumsum(torch.nn.functional.normalize(steps, dim=-1), dim=1)
    pts = pts - pts.mean(dim=1, keepdim=True)
    return pts / pts.flatten(1).std(dim=1).clamp_min(1e-6).reshape(B, 1, 1)


def _arch_branched(u: torch.Tensor, d_fractal: torch.Tensor) -> torch.Tensor:
    """Branched, aggregate-like cloud: point n sits at distance n^(1/d_f) in a random direction.

    d_f is the "fractal dimension": the cloud's mass grows like radius^d_f. Values of about 1.8
    to 2.6 are typical of aggregates. The result is centred and scaled so that its coordinates
    have standard deviation 1 (radius of gyration about 1.73).
    """
    B, P, _ = u.shape
    df = d_fractal.reshape(B, 1, 1).clamp(1.3, 3.0)
    n = torch.arange(1, P + 1, device=u.device, dtype=u.dtype).reshape(1, P, 1)
    radius = n ** (1.0 / df)
    pts = torch.nn.functional.normalize(u, dim=-1) * radius
    pts = pts - pts.mean(dim=1, keepdim=True)
    return pts / pts.flatten(1).std(dim=1).clamp_min(1e-6).reshape(B, 1, 1)


def _arch_sheet(u: torch.Tensor, thickness: torch.Tensor) -> torch.Tensor:
    """Flat sheet: x = (u1, u2, thickness * u3), then scaled so that its coordinates have standard
    deviation 1 (radius of gyration about 1.73).

    Stands for nanodiscs, bicelles or small flat membrane patches.
    """
    # Built without changing tensors in place, so that derivatives keep working.
    t = thickness.reshape(-1, 1).clamp(0.02, 1.0)     # (B, 1), to match u[..., 2] of shape (B, P)
    pts = torch.stack([u[..., 0], u[..., 1], u[..., 2] * t], dim=-1)
    return pts / pts.flatten(1).std(dim=1).clamp_min(1e-6).reshape(-1, 1, 1)


def sample_object_points(
    u: torch.Tensor,                 # (B, P, 3) the fixed random normal points
    topology: torch.Tensor,          # (B, 5) weights of the five shapes
    rg_A: torch.Tensor,              # (B,) radius of gyration, in Angstrom
    aniso: torch.Tensor,             # (B, 2) log-stretch along object axes 1 and 2
    hollowness: torch.Tensor,        # (B,)
    persistence: torch.Tensor,       # (B,)
    d_fractal: torch.Tensor,         # (B,)
    sheet_thickness: torch.Tensor,   # (B,)
) -> torch.Tensor:
    """Build random point clouds: mix the five shapes, stretch, and scale to the drawn size.

    The weights are divided by their sum, and the mix is done on point POSITIONS, so a mixed
    weight gives a genuinely in-between shape (not two shapes drawn on top of each other). Then
    two axes are stretched by exp(aniso), and the cloud is scaled to radius of gyration rg_A.
    Returns (B, P, 3) points in Angstrom, centred on 0 and not yet rotated.
    """
    parts = torch.stack([
        _arch_compact(u),
        _arch_shell(u, hollowness),
        _arch_filament(u, persistence),
        _arch_branched(u, d_fractal),
        _arch_sheet(u, sheet_thickness),
    ], dim=0)                                                            # (5, B, P, 3)
    w = topology.clamp_min(0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pts = (parts * w.T.reshape(N_ARCHETYPES, -1, 1, 1)).sum(dim=0)       # (B, P, 3)

    scale = torch.ones_like(pts)
    scale[..., 1] = torch.exp(aniso[:, 0]).reshape(-1, 1)
    scale[..., 2] = torch.exp(aniso[:, 1]).reshape(-1, 1)
    pts = pts * scale
    rg_now = pts.pow(2).sum(-1).mean(-1).sqrt().clamp_min(1e-6)          # (B,)
    return pts * (rg_A / rg_now).reshape(-1, 1, 1)


# --- pieces of the molecule (Eq. S8) ------------------------------------------------------------

def _n_keep_from_frac(frac, n_valid, min_atoms: int):
    """How many atoms a piece holding a fraction `frac` of the molecule keeps.

    ceil(frac * n_valid), but at least `min_atoms` (7 residues, about two turns of a helix;
    anything smaller is better described by a point cloud) and never more than the molecule has.
    frac = 1 always keeps every atom, so a "whole copy" really is whole. `n_valid` is counted per
    molecule, because molecules in one bank can have different numbers of atoms.
    """
    exact_full = frac >= 1.0
    n = torch.ceil(frac.clamp_min(0.0) * n_valid.to(frac.dtype)).long()
    n = torch.minimum(n.clamp_min(int(min_atoms)), n_valid)      # at least min_atoms, at most all atoms
    return torch.where(exact_full, n_valid, n)


def points_from_bank_ball(
    models: torch.Tensor,            # (M, R, 3, A_max) the bank
    model_idx: torch.Tensor,         # (B, S, 2) which structure each piece comes from
    seed_u: torch.Tensor,            # (B, S) in [0, 1): picks the atom the piece is cut around
    frac: torch.Tensor,              # (B, S) in (0, 1]: fraction of the atoms to keep
    min_atoms: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a compact piece out of the molecule: one atom plus its nearest neighbours in 3-D (Eq. S8).

    A piece cut as a ball (rather than as a stretch of the chain) is compact: its size grows
    like (fraction)^(1/3), as for a real broken-off piece. A stretch of the chain, by contrast,
    can wind through the whole molecule once it is long enough.
    Returns the piece's points (B, S, W, 3), centred and padded with inf, and the fraction of
    the molecule's atoms it holds (B, S).
    """
    B, S, _ = model_idx.shape
    sel = models[model_idx[..., 0].reshape(-1), model_idx[..., 1].reshape(-1)]    # (BS, 3, A_max)
    valid = torch.isfinite(sel).all(dim=1)                                        # (BS, A_max)
    n_valid = valid.sum(dim=1).clamp_min(1)

    n_keep = _n_keep_from_frac(frac.reshape(-1), n_valid, min_atoms)               # (BS,)
    seed = (seed_u.reshape(-1).clamp(0.0, 0.999) * n_valid.to(seed_u.dtype)).long()
    safe = torch.where(valid.unsqueeze(1), sel, torch.zeros_like(sel))             # inf would spoil the distances
    ctr0 = torch.gather(safe, 2, seed.reshape(-1, 1, 1).expand(-1, 3, 1))          # (BS, 3, 1) the seed atom
    d2 = (safe - ctr0).pow(2).sum(dim=1)                                           # (BS, A_max) squared distances
    d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))                 # padding sorts last

    W = int(n_keep.max().item())
    _, idx = torch.topk(d2, W, dim=1, largest=False)                               # nearest W atoms
    pts = torch.gather(safe, 2, idx.unsqueeze(1).expand(-1, 3, -1))                # (BS, 3, W)

    keep = torch.arange(W, device=sel.device).unsqueeze(0) < n_keep.unsqueeze(1)
    cnt = keep.sum(1).clamp_min(1).reshape(-1, 1, 1).to(pts.dtype)
    pts = pts - (pts * keep.unsqueeze(1)).sum(2, keepdim=True) / cnt               # centre the piece
    pts = torch.where(keep.unsqueeze(1), pts, torch.full_like(pts, float("inf")))

    mass = (n_keep.to(pts.dtype) / n_valid.to(pts.dtype)).reshape(B, S)
    return pts.permute(0, 2, 1).reshape(B, S, W, 3), mass


# --- straight-through random choices -------------------------------------------------------------
# A yes/no or pick-one-of-many draw has no useful derivative: a tiny change of a probability does
# not change the outcome. The "straight-through" trick (paper: Bengio et al. 2013; Jang et al.
# 2017) keeps the exact hard outcome for the image, but lets derivatives pass through a smooth
# version of it. The images themselves are exactly the same as with a plain random draw.

def st_gumbel_onehot(probs, u, tau: float = 0.5):
    """Pick one of K options with probabilities `probs`, as a one-hot vector (straight-through).

    The pick uses the "Gumbel-max" rule on the uniform numbers `u`, which gives exactly the
    requested probabilities. The returned value is the hard one-hot pick; its derivative is that
    of a softmax with temperature `tau`. Used for the number of clutter objects.
    """
    p = probs if torch.is_tensor(probs) else torch.as_tensor(probs, dtype=u.dtype, device=u.device)
    p = p.to(u.dtype).to(u.device).clamp_min(1e-9)
    logits = torch.log(p / p.sum(-1, keepdim=True))
    _u = u.clamp(1e-9, 1 - 1e-9)
    g = -torch.log(-torch.log(_u))                       # Gumbel noise from uniform numbers
    y = logits + g
    hard = torch.nn.functional.one_hot(y.argmax(-1), p.shape[-1]).to(u.dtype)
    soft = torch.softmax(y / tau, dim=-1)
    return hard + (soft - soft.detach())


def st_bernoulli(p, u, tau: float = 0.1):
    """A yes/no draw with probability `p` (1 = yes), straight-through.

    The value is exactly (u < p) for the given uniform numbers `u`, so images do not change. The
    derivative with respect to p comes from the smooth version
    sigmoid((logit(p) - logit(u)) / tau). A smaller tau follows the hard step more closely.
    """
    if not torch.is_tensor(p):
        p = torch.as_tensor(float(p), device=u.device, dtype=u.dtype)
    p = p.clamp(1e-6, 1 - 1e-6).expand_as(u)
    uu = u.clamp(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((torch.logit(p) - torch.logit(uu)) / tau)
    hard = (u < p).to(u.dtype)
    return hard + (soft - soft.detach())


# --- drawing the clutter --------------------------------------------------------------------------

def _render_group(out, pts, quat, offset, contrast, present, sigma, active,
                  num_pixels, pixel_size):
    """Rotate, place and draw a group of objects, then add each to its image.

    Only the (image, slot) pairs marked in `active` are drawn; empty slots would add exactly zero
    anyway. Every object's points share its weight `contrast` equally, together with the same
    1 / (2 pi sigma^2) factor as the molecule, so an object of weight 1 is as bright, in total,
    as the molecule.
    """
    idx = active.nonzero(as_tuple=False)                       # (K, 2) -> (image, slot)
    if idx.numel() == 0:
        return out
    b_i, s_i = idx[:, 0], idx[:, 1]
    K, P = int(idx.shape[0]), int(pts.shape[2])

    R = quaternions_to_rotation_matrices(quat[b_i, s_i])                       # (K, 3, 3)
    q = torch.einsum("kij,kpj->kpi", R, pts[b_i, s_i])                         # (K, P, 3) rotated
    q = torch.cat([q[..., :2] + offset[b_i, s_i].unsqueeze(1), q[..., 2:]], dim=-1)   # moved in x, y
    coords = q.permute(0, 2, 1)                                                # (K, 3, P)

    sg = sigma[b_i]                                                            # (K,)
    fin = torch.isfinite(coords).all(dim=1)
    nval = fin.sum(dim=-1).clamp_min(1).to(out.dtype)
    w = ((contrast[b_i, s_i] * present[b_i, s_i]) / nval
         / (2.0 * torch.pi * sg ** 2)).unsqueeze(-1).expand(K, P)
    proj = project_points(coords, w, sg.reshape(K, 1, 1), num_pixels, pixel_size)
    return out.index_add(0, b_i, proj.to(out.dtype))


def render_v3_clutter(base_signal, models, p: dict, main_sigma, num_pixels, pixel_size):
    """Stage 2: add the clutter objects to the clean projection, BEFORE the microscope blur.

    `base_signal` (B, N, N) is the molecule's projection; `p` holds the clutter part of theta
    (built by to_v3_block); `main_sigma` is each image's bead width, shared by its clutter.

    Each object is "t times a piece of the molecule, plus (1 - t) times a random point cloud",
    at one position and orientation:
      - the piece's weight is t times its brightness, which is either the fraction of the
        molecule's atoms it holds (about t, so the weight is about t * t) or, if cl_use_free is
        1, the freely drawn cl_amp_free;
      - the cloud's weight is (1 - t) times its contrast, which may be negative.
    t = 1 draws only the copy, t = 0 only the cloud. Returns (B, N, N).

    The piece is cut with a hard edge (points_from_bank_ball), so derivatives with respect to t
    come from the weights above, not from the size of the piece.
    """
    B = base_signal.shape[0]
    S = int(p["n_slots"]); P = int(p["n_points_proc"])
    sig = main_sigma.reshape(B)
    out = base_signal
    present = p["present"]

    t = p["t_pdb"]

    act_bank = (present > 0) & (t > 0)
    if bool(act_bank.any()):
        min_at = int(p.get("min_fragment_atoms", 7))
        bpts, mass = points_from_bank_ball(models, p["model_idx"], p["win_start"],
                                           t, min_atoms=min_at)
        # brightness: the free amplitude when use_free is 1, otherwise the fraction of atoms
        g = p["use_free"]
        g = g.to(mass.dtype) if g.dtype != mass.dtype else g
        amp = g * p["amp_free"] + (1.0 - g) * mass
        out = _render_group(out, bpts, p["quat"], p["offset_A"], amp * t, present, sig,
                            act_bank, num_pixels, pixel_size)

    act_proc = (present > 0) & (t < 1)
    if bool(act_proc.any()):
        flat = lambda x, *tail: x.reshape(B * S, *tail)
        ppts = sample_object_points(
            flat(p["u"], P, 3), flat(p["topology"], N_ARCHETYPES), flat(p["rg_A"]),
            flat(p["aniso"], 2), flat(p["hollowness"]), flat(p["persistence"]),
            flat(p["d_fractal"]), flat(p["sheet_thickness"])).reshape(B, S, P, 3)
        out = _render_group(out, ppts, p["quat"], p["offset_A"],
                            p["contrast_proc"] * (1.0 - t), present, sig,
                            act_proc, num_pixels, pixel_size)
    return out


# ==============================================================================================
# PART 11 -- STAGE 4: THE NOISE MIXTURE  (paper: Eq. S10, S11, S17)
# ==============================================================================================
# Three noise fields, each scaled to variance 1, are mixed with the image's weights
# (theta["noise_w"]), the mix is rescaled to mean 0 and variance 1 again, and it is added to the
# clean image with the strength sigma_n found from the requested SNR (Eq. S15, S17):
#
#     noisy image = clean image + sigma_n * normalise(w_col * coloured + w_shot * shot + w_white * white)
#
# Because the mix always has variance exactly 1, the weights only change the TEXTURE of the
# noise, and sigma_n alone sets how strong it is.
#
# Where the coloured noise gets its spectrum, from best to worst:
#   1. the data set's own measured spectrum: a file given as `empirical_cache_file`, or measured
#      from MRC files under `empirical_root_folder` (this is what the paper uses);
#   2. an average spectrum over several data sets, read from _GLOBAL_AVG_PSD_FILE if it exists;
#   3. a simple 1 / r^0.39 formula.
# The fallback is silent, so check that your spectrum file is found.

# gamma of Eq. S10: the shot-noise dose scale is lambda = clip(gamma * SNR, 1, 200).
POISSON_GAMMA = 43.0


def _resolve_empirical_psd(
    *,
    empirical_cache_file: str | None,
    empirical_root_folder: str | None,
    empirical_num_images: int | None,
    empirical_selection_mode: str,
    empirical_random_seed: int,
    empirical_batch_size: int,
    strict: bool = False,
) -> torch.Tensor | None:
    """Tier 1: the data set's own spectrum, from a file or measured from MRC files.

    Returns None when nothing was configured, or when the file is missing and `strict` is False,
    so that the caller can fall back to tier 2. With strict=True a missing file raises an error.
    """
    if empirical_cache_file is None and empirical_root_folder is None:
        return None
    try:
        stats = get_empirical_noise_parameters_cached(
            root_folder=empirical_root_folder,
            num_images=empirical_num_images,
            selection_mode=empirical_selection_mode,
            random_seed=empirical_random_seed,
            batch_size=empirical_batch_size,
            device="cpu",
            cache_file=empirical_cache_file,
        )
        return stats["mean_psd"]
    except FileNotFoundError:
        if strict:
            raise
        return None


def _synthetic_power_law_psd(
    num_pixels: int,
    alpha: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Tier 3: a simple (N, N) spectrum 1 / (r^alpha + 1e-6), in FFT order, divided by its mean."""
    freq = torch.fft.fftfreq(num_pixels, device=device)
    fy, fx = torch.meshgrid(freq, freq, indexing="ij")
    r = torch.sqrt(fx**2 + fy**2)
    psd = 1.0 / (r**alpha + 1e-6)
    return psd / psd.mean()


# Tier 2: a file with the average radial spectrum over several data sets: a dictionary saved with
# torch.save, holding "common_freq" (1-D, frequencies in cycles per pixel, increasing) and
# "avg_radial" (the spectrum at those frequencies, same length). It is looked for in a folder
# called "data" next to v3_core.py, as data/global_average_empirical_psd.pt. It is not part of
# v3_core.py; if it is missing, tier 3 is used.
_GLOBAL_AVG_PSD_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "global_average_empirical_psd.pt"
)


_GLOBAL_AVG_PSD_DATA: tuple[torch.Tensor, torch.Tensor] | None = None


def _load_global_avg_radial_psd() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read the tier-2 average spectrum file once; return None if it is not there."""
    global _GLOBAL_AVG_PSD_DATA
    if _GLOBAL_AVG_PSD_DATA is not None:
        return _GLOBAL_AVG_PSD_DATA
    if not _GLOBAL_AVG_PSD_FILE.is_file():
        return None
    try:
        d = torch.load(_GLOBAL_AVG_PSD_FILE, map_location="cpu", weights_only=False)
        _GLOBAL_AVG_PSD_DATA = (d["common_freq"].float(), d["avg_radial"].float())
    except Exception:
        return None
    return _GLOBAL_AVG_PSD_DATA


def _global_average_psd(
    num_pixels: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor | None:
    """Tier 2: turn the average radial spectrum into an (N, N) spectrum, or None if it is missing.

    The radial values are interpolated linearly at each pixel's frequency (as a fraction of the
    sampling rate), and the result is divided by its mean.
    """
    data = _load_global_avg_radial_psd()
    if data is None:
        return None
    common_freq, avg_radial = data
    common_freq = common_freq.to(device)
    avg_radial = avg_radial.to(device)

    freq = torch.fft.fftfreq(num_pixels, device=device)
    fy, fx = torch.meshgrid(freq, freq, indexing="ij")
    r = torch.sqrt(fx**2 + fy**2).flatten()  # (N*N,)

    # linear interpolation of avg_radial(common_freq) at every pixel's frequency r
    idx_right = torch.searchsorted(common_freq, r).clamp(1, len(common_freq) - 1)
    idx_left = idx_right - 1
    fl = common_freq[idx_left]; fr = common_freq[idx_right]
    pl = avg_radial[idx_left]; pr = avg_radial[idx_right]
    t = ((r - fl) / (fr - fl).clamp_min(1e-12)).clamp(0.0, 1.0)
    psd = (pl + t * (pr - pl)).reshape(num_pixels, num_pixels)
    return psd / psd.mean()


def build_mixture_noise(
    image: torch.Tensor,
    snr_for_dose: torch.Tensor,
    weights: torch.Tensor,
    *,
    parameter_source: str = "standard",
    psd_exponent: float = DEFAULT_SYNTHETIC_PSD_EXPONENT,
    poisson_dose_scale: float = POISSON_GAMMA,
    empirical_cache_file: str | None = None,
    empirical_root_folder: str | None = None,
    empirical_num_images: int | None = 10000,
    empirical_selection_mode: str = "random",
    empirical_random_seed: int = 42,
    empirical_batch_size: int = 64,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the mixed noise field (Eq. S17): mean 0 and variance exactly 1 in every image.

    Inputs
        image         (B, N, N) clean images after the CTF, clutter included. The shot noise
                      follows their brightness.
        snr_for_dose  (B, 1, 1) the LINEAR SNR, used only for the shot-noise dose (Eq. S10).
        weights       (B, 3) weights of the coloured, shot and white fields: theta["noise_w"].
        the empirical_* and parameter_source keywords say where the spectrum comes from (see the
        top of this part); parameter_source="empirical" makes a missing file an error.
        seed          leave it None, as render does. If a number is given, EACH of the three
                      fields restarts the random stream from it, so the coloured and white fields
                      are built from the same random numbers and are strongly correlated.
    Output
        the noise field (B, N, N), and a dictionary with the separate fields and the spectrum
        (useful for checks and plots).

    The three fields are drawn from PyTorch's global random stream, in the fixed order coloured,
    shot, white; `render` seeds that stream. On the CPU, the shot-noise sampler uses an amount of
    the stream that depends on the doses. Changing one image's dose therefore changes the shot
    noise of the images after it and the white noise of every image in the batch (with the same
    statistics).
    """
    if seed is not None:
        torch.manual_seed(seed)
    B, N, _ = image.shape
    if weights.shape != (B, 3):
        raise ValueError(f"weights must be (B, 3) in {NOISE_CHANNELS} order, got {tuple(weights.shape)}")

    # the spectrum: tier 1, else tier 2, else tier 3
    empirical_psd = _resolve_empirical_psd(
        empirical_cache_file=empirical_cache_file,
        empirical_root_folder=empirical_root_folder,
        empirical_num_images=empirical_num_images,
        empirical_selection_mode=empirical_selection_mode,
        empirical_random_seed=empirical_random_seed,
        empirical_batch_size=empirical_batch_size,
        strict=(parameter_source == "empirical"),
    )
    psd = empirical_psd
    if psd is None:
        psd = _global_average_psd(N, device=image.device)                                # tier 2
    if psd is None:
        psd = _synthetic_power_law_psd(N, float(psd_exponent), device=image.device)      # tier 3

    # Only one of the three fields is coloured; colouring all three would make them alike.
    noise_fft_slot = colored_fft_noise_from_psd(image, psd, seed=seed)
    noise_poisson = _poisson_noise_unit(image, snr_for_dose, dose_scale=poisson_dose_scale, seed=seed)
    noise_gauss = _gaussian_noise_unit(image, seed=seed)

    w = weights.to(device=image.device, dtype=image.dtype)
    w_fft, w_poisson, w_gauss = w[:, 0], w[:, 1], w[:, 2]
    noise = (
        w_fft.view(B, 1, 1) * noise_fft_slot
        + w_poisson.view(B, 1, 1) * noise_poisson
        + w_gauss.view(B, 1, 1) * noise_gauss
    )
    # Rescale the SUM (not the weights): this makes the mix exactly variance 1 in every image.
    noise = noise - noise.mean(dim=(-2, -1), keepdim=True)
    noise = noise / noise.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    debug = {
        "noise_fft_slot": noise_fft_slot, "noise_poisson": noise_poisson, "noise_gauss": noise_gauss,
        "empirical_psd": empirical_psd, "psd": psd,
        "mix_w_fft": w_fft, "mix_w_poisson": w_poisson, "mix_w_gaussian": w_gauss,
    }
    return noise, debug


def add_noise(
    image: torch.Tensor,
    snr_for_dose: torch.Tensor,
    weights: torch.Tensor,
    *,
    noise_scale: torch.Tensor,
    return_debug: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Stage 4: return image + noise_scale * (mixed noise), as in Eq. S17.

    `noise_scale` is sigma_n, measured by the renderer on the molecule alone (Eq. S15), so the
    requested SNR does not depend on how much clutter is around it. Other keywords are passed to
    build_mixture_noise (where the spectrum comes from).
    """
    noise, debug = build_mixture_noise(image, snr_for_dose, weights, **kwargs)
    noise_power = noise_scale.reshape(-1, 1, 1)
    noisy = image + noise * noise_power
    if return_debug:
        debug["noise_power"] = noise_power
        return noisy, debug
    return noisy


# ==============================================================================================
# PART 12 -- THE SAMPLER: phi -> theta  (paper: Eq. S13 and the table in SI S3)
# ==============================================================================================
# sample_theta draws the settings of n images from the prior phi. Every random number is drawn
# first, independently of phi, and then pushed through a formula that uses phi
# ("reparameterisation", Eq. S13). That is why derivatives of theta with respect to phi exist.

_SEED_MODULUS = 1 << 24   # per-image seeds are below 2^24, so they are stored exactly as float32


# The white-only noise regime keeps a probability of at least 2 %, as long as REGIME_full_psd
# stays below 0.98 (see REGIME_white in Part 7).
WHITE_REGIME_FLOOR = 0.02


PROB_FLOORS = {"REGIME_white": WHITE_REGIME_FLOOR}


# Lower bounds that the lower end of these ranges can never cross (values must be positive).
RANGE_LOWER_BOUNDS = {
    "SIGMA": 0.0,
    "DEFOCUS": 0.0,
    "B_FACTOR": 0.0,
    "rg_range_A": 0.0,
    "barcode_width_frac": 0.0,
    "MIX_RANGE_gaussian": 0.0,
    "MIX_RANGE_fft": 0.0,
    "MIX_RANGE_poisson": 0.0,
    "amp_free_range": 0.0,
}


# Upper bounds for ranges that cannot go above them (noise weights stay at most 1).
RANGE_UPPER_BOUNDS = {
    "MIX_RANGE_gaussian": 1.0,
    "MIX_RANGE_fft": 1.0,
    "MIX_RANGE_poisson": 1.0,
}


RANGE_FLOOR_EPS = 1e-12


@dataclass(frozen=True)
class RenderShape:
    """Fixed facts about one data set (box, pixels, noise spectrum). Nothing random lives here.

    Fields you usually set:
        num_pixels   N, the box size in pixels. With a measured spectrum, N must equal the box
                     size of the particles it was measured on.
        pixel_size   Angstrom per pixel. Use your particles' pixel size: it is not read from
                     the spectrum or the MRC files.
        noise        where the coloured-noise spectrum comes from (Part 11), as a dictionary:
                       {"empirical_cache_file": "my_dataset_psd.pt"}   a saved spectrum, or
                       {"empirical_root_folder": "particles/"}         measure it from MRC files
                     plus, optionally, "empirical_num_images" (10000), "empirical_random_seed"
                     (42), "empirical_selection_mode" ("random"), "empirical_batch_size" (64), and
                     "parameter_source": "empirical" to make a missing file an error.
                     An empty dictionary means: use the fallback spectra.
    Fields you usually leave alone:
        n_slots      must stay 5: theta has room for exactly 5 clutter objects;
        n_points_proc, min_fragment_atoms   clutter model sizes (Part 10);
        amp          amplitude contrast of the CTF (0.1);
        shift_A      largest random sideways shift of the molecule, in Angstrom: each image is
                     shifted by a uniform amount in [-shift_A, +shift_A] along x and along y
                     (0 in the paper: no shift);
        count_tau, gate_tau, side_tau                smoothness of the straight-through draws;
        r_inner_A    radius used only to record whether a clutter object overlaps the molecule;
                     filled in by with_bank;
        noise_floor_std   None by default. If set, an image whose blurred molecule is empty
                     (standard deviation below 1e-6) gets the noise level
                     noise_floor_std / sqrt(SNR) instead of 0, and non-finite pixel values are
                     set to 0.

    Call shape.with_bank(bank) once you have a bank; sample_theta and render_from_phi do it for you.
    """

    num_pixels: int
    pixel_size: float

    # clutter model sizes
    n_slots: int = N_SLOTS
    n_points_proc: int = int(_V3_DEFAULTS["n_points_proc"])
    min_fragment_atoms: int = int(_V3_DEFAULTS["min_fragment_atoms"])

    # physical constants of the paper's set-up (not priors, so not in phi)
    amp: float = 0.1                  # amplitude contrast
    shift_A: float = 0.0              # no sideways shift

    # smoothness of the straight-through draws (they change derivatives only, never images)
    count_tau: float = float(_V3_DEFAULTS["count_tau"])
    gate_tau: float = float(_V3_DEFAULTS["gate_tau"])
    side_tau: float = 0.5             # for the side of the edge band

    # computed from the bank by with_bank (only used for a diagnostic, never for the image)
    r_inner_A: float | None = None

    # where the coloured-noise spectrum comes from (see the docstring above)
    noise: dict = field(default_factory=dict)

    noise_floor_std: float | None = None

    @property
    def box_A(self) -> float:
        """The box width in Angstrom: N times the pixel size."""
        return float(self.num_pixels) * float(self.pixel_size)

    def with_bank(self, bank: torch.Tensor) -> "RenderShape":
        """Return a copy with r_inner_A filled in from the bank: min(0.8 x molecule radius, 0.45 x half box)."""
        if self.r_inner_A is not None:
            return self
        envelope = bank_envelope_A(bank) * 0.8
        half = 0.5 * self.box_A
        return replace(self, r_inner_A=float(min(envelope, 0.45 * half)))


def _theta_shapes(n_slots: int) -> dict[str, tuple[int, ...]]:
    """The natural shape of every theta slot for one image (used by unpack_theta)."""
    S = n_slots
    return {
        "seed": (), "conformer_idx": (2,), "quaternion": (4,), "sigma": (),
        "shift": (2,), "defocus": (), "b_factor": (), "amp": (), "log10_snr": (),
        "noise_w": (3,), "noise_regime": (), "noise_white_flag": (),
        "band_present": (), "band_width_frac": (), "band_side": (),
        "band_side_onehot": (N_SIDES,),
        "cl_present": (S,), "cl_t_pdb": (S,), "cl_win_start": (S,),
        "cl_model_idx": (S, 2), "cl_contrast_proc": (S,), "cl_amp_free": (S,),
        "cl_use_free": (S,), "cl_topology": (S, N_ARCHETYPES), "cl_rg_A": (S,),
        "cl_aniso": (S, 2), "cl_hollowness": (S,), "cl_persistence": (S,),
        "cl_d_fractal": (S,), "cl_sheet_thick": (S,), "cl_quat": (S, 4),
        "cl_offset_A": (S, 2),
    }


def unpack_theta(theta: torch.Tensor, n_slots: int = N_SLOTS) -> dict[str, torch.Tensor]:
    """Split theta (B, 156) into a dictionary of named pieces, e.g. out["defocus"] has shape (B,)."""
    if theta.ndim != 2 or theta.shape[1] != THETA_LEN:
        raise ValueError(f"theta must be (B, {THETA_LEN}), got {tuple(theta.shape)}")
    B = theta.shape[0]
    shapes = _theta_shapes(n_slots)
    out, off = {}, 0
    for s in THETA_LAYOUT:
        chunk = theta[:, off:off + s.size]
        out[s.name] = chunk.reshape(B, *shapes[s.name])
        off += s.size
    return out


def pack_theta(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    """The reverse of unpack_theta: join named pieces back into theta (B, 156)."""
    B = next(iter(fields.values())).shape[0]
    parts = []
    for s in THETA_LAYOUT:
        v = fields[s.name]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        parts.append(v.reshape(B, -1).to(torch.float32))
    theta = torch.cat(parts, dim=1)
    if theta.shape[1] != THETA_LEN:
        raise ValueError(f"packed width {theta.shape[1]} != THETA_LEN {THETA_LEN}")
    return theta


def derive_u(seed, n_slots: int, n_points: int, device="cpu") -> torch.Tensor:
    """Each image's fixed random normal points for its clutter clouds, rebuilt from its seed.

    (B,) seeds -> (B, n_slots, n_points, 3). These 5 x 1024 x 3 = 15,360 numbers per image are
    the "fixed base randomness" of the point clouds (SI S3); they are not stored in theta, only
    their seed is. Each image uses its own private random generator on the CPU, so the result
    depends neither on the batch nor on the device, and it does not disturb any other random
    numbers.
    """
    seeds = torch.as_tensor(seed).reshape(-1).detach().cpu()
    out = []
    for s in seeds.tolist():
        g = torch.Generator(device="cpu").manual_seed(int(s) % (1 << 63))
        out.append(torch.randn(n_slots, n_points, 3, generator=g))
    return torch.stack(out).to(device)


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """The inverse of softplus: log(exp(x) - 1)."""
    return torch.log(torch.expm1(x))


def unpack_phi(phi: torch.Tensor) -> dict[str, torch.Tensor]:
    """Turn the stored phi (44 numbers) into a dictionary of PHYSICAL values.

    Probabilities come out between 0 and 1, positive numbers stay positive, and every range
    comes out as [lo, hi] with hi > lo, whatever the stored numbers are. The keys are the
    names in PHI_LAYOUT (Part 7).
    """
    if phi.reshape(-1).shape[0] != PHI_LEN:
        raise ValueError(f"phi must have {PHI_LEN} entries, got {phi.numel()}")
    phi = phi.reshape(-1)
    out, off = {}, 0
    for s in PHI_LAYOUT:
        raw = phi[off:off + s.size]
        off += s.size
        if s.kind == "logit":
            v = torch.sigmoid(raw)
            out[s.name] = v if s.size > 1 else v[0]
        elif s.kind == "logit_floor":
            # p = f + (1 - f) * sigmoid(raw), which is never below the floor f
            f = PROB_FLOORS[s.name]
            v = f + (1.0 - f) * torch.sigmoid(raw)
            out[s.name] = v if s.size > 1 else v[0]
        elif s.kind == "softplus":
            v = F.softplus(raw)
            out[s.name] = v if s.size > 1 else v[0]
        elif s.kind == "ordered":
            lo = raw[0]
            bound = RANGE_LOWER_BOUNDS.get(s.name)
            if bound is not None:
                # the lower end is kept above its bound (softplus is always positive)
                lo = bound + F.softplus(lo)
            hi = lo + torch.exp(raw[1])
            top = RANGE_UPPER_BOUNDS.get(s.name)
            if top is not None:
                # Keep hi (and lo) at most `top`. The clamp is applied to the value only; the
                # derivative passes through unchanged ("straight-through"). The expression is
                # written as clamped.detach() + (x - x.detach()) on purpose: it stays exact for
                # very large x.
                lo = lo.clamp(max=top - RANGE_FLOOR_EPS).detach() + (lo - lo.detach())
                hi = hi.clamp(max=top).detach() + (hi - hi.detach())
            out[s.name] = torch.stack([lo, hi])
        elif s.kind == "simplex":
            out[s.name] = torch.softmax(raw, dim=0)
        else:
            raise ValueError(f"unknown phi kind {s.kind!r}")
    return out


def pack_phi(values: dict, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Turn a dictionary of PHYSICAL prior values into the stored phi vector (44 numbers).

    The dictionary needs every name of PHI_LAYOUT. Ranges are [lo, hi] lists, probabilities are
    numbers between 0 and 1, and count_probs / side_weights are lists that add up to 1 (they are
    divided by their sum anyway). SNR ranges are in log10. Part 14 has a complete example
    (the paper's Table S3). The reverse of unpack_phi.
    """
    parts = []
    for s in PHI_LAYOUT:
        v = torch.as_tensor(values[s.name], dtype=dtype, device=device).reshape(-1)
        if v.numel() != s.size:
            raise ValueError(f"{s.name}: expected {s.size} values, got {v.numel()}")
        if s.kind == "logit":
            parts.append(torch.logit(v.clamp(1e-12, 1 - 1e-12)))
        elif s.kind == "logit_floor":
            f = PROB_FLOORS[s.name]
            q = ((v - f) / (1.0 - f)).clamp(1e-12, 1 - 1e-12)
            parts.append(torch.logit(q))
        elif s.kind == "softplus":
            parts.append(_softplus_inv(v.clamp_min(1e-12)))
        elif s.kind == "ordered":
            lo, hi = v[0], v[1]
            bound = RANGE_LOWER_BOUNDS.get(s.name)
            top = RANGE_UPPER_BOUNDS.get(s.name)
            if top is not None:
                lo = lo.clamp(max=top - RANGE_FLOOR_EPS)
                hi = hi.clamp(max=top)
            lo_raw = (lo if bound is None
                      else _softplus_inv((lo - bound).clamp_min(RANGE_FLOOR_EPS)))
            parts.append(torch.stack([lo_raw, torch.log((hi - lo).clamp_min(1e-12))]))
        elif s.kind == "simplex":
            parts.append(torch.log(v.clamp_min(1e-12) / v.clamp_min(1e-12).sum()))
        else:
            raise ValueError(f"unknown phi kind {s.kind!r}")
    return torch.cat(parts)


def _kumaraswamy_inv(u, a, b):
    """Turn a uniform number u in [0, 1] into a Kumaraswamy(a, b) number in [0, 1].

    x = (1 - (1 - u)^(1/b))^(1/a). With a = b = 1, x = u. Used for the edge-band depth.
    """
    return (1.0 - (1.0 - u).clamp_min(1e-12) ** (1.0 / b)).clamp(1e-12, 1.0) ** (1.0 / a)


def sample_theta(
    phi: torch.Tensor,
    n: int,
    seed: int,
    bank: torch.Tensor,
    cfg: RenderShape,
    *,
    dtype=torch.float32,
    return_diagnostics: bool = False,
):
    """Draw the settings of `n` images from the prior `phi`. Derivatives with respect to phi exist.

    Inputs
        phi    the stored prior (44 numbers), e.g. pack_phi(DEFAULT_PRIOR).
        n      how many images.
        seed   makes the draw repeatable: the same seed gives the same theta.
        bank   the conformer bank (M, R, 3, A).
        cfg    the RenderShape (the box size matters for where clutter can sit).
    Output
        theta (n, 156); with return_diagnostics=True also a dictionary holding "cl_on_target"
        (1 where a clutter object landed on the molecule; for information only).

    How each part of theta is drawn (every draw uses its own uniform or normal random numbers):
      - optics: the conformer and the viewing direction uniformly; sigma, defocus, B-factor and
        log10 SNR uniformly inside their ranges;
      - noise: three weights, then one of three regimes: coloured only (weights 1, 0, 0) with
        probability REGIME_full_psd, white only (0, 0, 1) with probability REGIME_white (or
        1 - REGIME_full_psd, if that is smaller), and the mix of the three weights otherwise;
      - edge band: present with probability barcode_prob, then a depth (Kumaraswamy shape inside
        barcode_width_frac) and a side (side_weights);
      - clutter: the number of objects (count_probs), then for each object its type t, piece
        position, contrast, brightness, shape weights and shape numbers, size, stretch,
        orientation and position (Eq. S7). The shape numbers have FIXED ranges that are not in
        phi: hollowness 0-1, persistence 0.70-0.99, fractal dimension 1.5-2.7, sheet
        thickness 0.05-0.45, and shape weights 0-1 each.
    The yes/no and pick-one draws:
      - the "mostly white" branch, the noise regime and the free brightness use st_bernoulli,
        and the number of objects uses st_gumbel_onehot (both in Part 10);
      - the band side uses the same Gumbel-softmax trick, written out in place.
    These are straight-through draws: the images are exactly those of plain random draws, while
    derivatives still reach the probabilities. Two choices are plain cut-offs on one uniform
    number instead: whether there is a band, and whether t is exactly 0 or 1. Their
    probabilities get derivatives only through the band depth and through values of t strictly
    between 0 and 1.
    """
    cfg = cfg.with_bank(bank)
    models = as_4d(bank)
    M, R = int(models.shape[0]), int(models.shape[1])
    S = cfg.n_slots
    p = unpack_phi(phi)
    dev = phi.device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    rnd = lambda *sh: torch.rand(*sh, generator=g).to(dev)       # uniform numbers in [0, 1)
    rndn = lambda *sh: torch.randn(*sh, generator=g).to(dev)     # standard normal numbers

    def _rng(name, u):
        """lo + (hi - lo) * u for the range called `name`: uniform inside the range."""
        lo, hi = p[name][0], p[name][1]
        return lo + (hi - lo) * u

    # ---- optics ----------------------------------------------------------
    conformer_idx = torch.stack([
        torch.randint(0, M, (n,), generator=g),
        torch.randint(0, max(R, 1), (n,), generator=g),
    ], dim=1).to(dev).to(torch.float64)
    quaternion = rndn(n, 4)
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sigma = _rng("SIGMA", rnd(n))
    shift = (rnd(n, 2) * 2.0 - 1.0) * float(cfg.shift_A)
    defocus = _rng("DEFOCUS", rnd(n))
    b_factor = _rng("B_FACTOR", rnd(n))
    amp = torch.full((n,), float(cfg.amp), dtype=torch.float64, device=dev)
    log10_snr = _rng("SNR", rnd(n))

    # ---- noise weights ---------------------------------------------------
    w = {}
    white_flag = torch.zeros(n, dtype=torch.float64, device=dev)
    for c in NOISE_CHANNELS:
        u = rnd(n)
        lo, hi = p[f"MIX_RANGE_{c}"][0], p[f"MIX_RANGE_{c}"][1]
        a = p["CONC_fft"] if c == "fft" else torch.ones((), dtype=lo.dtype, device=dev)
        wc = lo + (hi - lo) * u ** (1.0 / a)
        if c == "fft":
            # with probability WHITE_FRAC_fft, draw the coloured weight from the bottom quarter
            # of its range instead, so some images are nearly white even in the mixed regime
            flag = st_bernoulli(p["WHITE_FRAC_fft"], rnd(n), float(cfg.gate_tau))
            w_white = lo + rnd(n) * 0.25 * (hi - lo)
            wc = flag * w_white + (1.0 - flag) * wc
            white_flag = flag
        w[c] = wc

    # the regime: one uniform number r decides; r < p_psd -> coloured only,
    # p_psd <= r < p_psd + p_white -> white only, otherwise the mix
    r = rnd(n)
    p_psd, p_wht = p["REGIME_full_psd"], p["REGIME_white"]
    g_psd = st_bernoulli(p_psd, r, float(cfg.gate_tau))
    g_cum = st_bernoulli((p_psd + p_wht).clamp(1e-6, 1 - 1e-6), r, float(cfg.gate_tau))
    g_white = g_cum - g_psd
    g_bridge = 1.0 - g_cum
    noise_w = torch.stack([
        g_psd + g_bridge * w["fft"],
        g_bridge * w["poisson"],
        g_white + g_bridge * w["gaussian"],
    ], dim=1)
    regime = torch.where(g_psd > 0.5, torch.full_like(r, float(REGIME_FULL_PSD)),
                         torch.where(g_white > 0.5, torch.full_like(r, float(REGIME_WHITE)),
                                     torch.full_like(r, float(REGIME_BRIDGE))))

    # ---- edge band -------------------------------------------------------
    # one uniform number: below 1 - barcode_prob there is no band (depth exactly 0); above it,
    # the rest of the interval is stretched to [0, 1] and turned into a depth
    ub = rnd(n)
    prob = p["barcode_prob"]
    x = (ub - (1.0 - prob)).clamp_min(0.0) / prob
    blo, bhi = p["barcode_width_frac"][0], p["barcode_width_frac"][1]
    depth = torch.where(
        x > 0,
        blo + (bhi - blo) * _kumaraswamy_inv(x, p["depth_a"], p["depth_b"]),
        torch.zeros_like(x),
    )
    # the side: a Gumbel-softmax pick among the four sides, straight-through
    logits = torch.log(p["side_weights"].clamp_min(1e-12)).expand(n, N_SIDES)
    _u = rnd(n, N_SIDES).clamp(1e-12, 1.0 - 1e-7)
    y_soft = torch.softmax((logits + (-torch.log(-torch.log(_u)))) / float(cfg.side_tau), dim=-1)
    side_idx = y_soft.argmax(dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, side_idx.unsqueeze(-1), 1.0)
    side_onehot = y_hard + (y_soft - y_soft.detach())

    # ---- clutter ---------------------------------------------------------
    # the type t of each object, from one uniform number (Eq. S8 and around it):
    # P(t = 0) = 1 - p_bank (a pure cloud), P(t = 1) = p_duplicate (a whole copy),
    # and t uniform in between otherwise
    p_dup, p_bank = p["p_duplicate"], p["p_bank"]
    p_proc = 1.0 - p_bank
    slab = (1.0 - p_proc - p_dup).clamp_min(1e-6)
    t_pdb = ((rnd(n, S) - p_proc) / slab).clamp(0.0, 1.0)
    win_start = rnd(n, S)
    contrast_proc = _rng("contrast_range", rnd(n, S))
    amp_free = _rng("amp_free_range", rnd(n, S))
    use_free = st_bernoulli(p["p_free_amp"], rnd(n, S), float(cfg.gate_tau))

    # where each object sits (Eq. S7): a random angle, and a distance r = r_out * u^(1/(2a)),
    # with r_out half the box diagonal so that the corners can be reached
    ang = rnd(n, S) * 2.0 * torch.pi
    r_out = 0.5 * (2.0 ** 0.5) * cfg.box_A
    expo = 1.0 / (2.0 * p["radial_exponent"])
    rr = r_out * rnd(n, S).clamp_min(1e-12) ** expo
    on_target_diag = (rr < float(cfg.r_inner_A)).to(rr.dtype)   # for information only

    # how many objects: a straight-through pick of N in 0..5; the first N slots are then "present"
    onehot = st_gumbel_onehot(p["count_probs"], rnd(n, S + 1), float(cfg.count_tau))
    present = onehot.flip(-1).cumsum(-1).flip(-1)[:, 1:]

    aw = p["aniso_log_halfwidth"]
    cl_quat = rndn(n, S, 4)
    cl_quat = cl_quat / cl_quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    seeds = torch.randint(0, _SEED_MODULUS, (n,), generator=g).to(dev).to(torch.float64)

    fields = {
        "seed": seeds.reshape(n, 1),
        "conformer_idx": conformer_idx,
        "quaternion": quaternion,
        "sigma": sigma.reshape(n, 1),
        "shift": shift,
        "defocus": defocus.reshape(n, 1),
        "b_factor": b_factor.reshape(n, 1),
        "amp": amp.reshape(n, 1),
        "log10_snr": log10_snr.reshape(n, 1),
        "noise_w": noise_w,
        "noise_regime": regime.reshape(n, 1),
        "noise_white_flag": white_flag.reshape(n, 1),
        "band_present": (depth > 0).to(depth.dtype).reshape(n, 1),
        "band_width_frac": depth.reshape(n, 1),
        "band_side": side_idx.to(depth.dtype).reshape(n, 1),
        "band_side_onehot": side_onehot,
        "cl_present": present,
        "cl_t_pdb": t_pdb,
        "cl_win_start": win_start,
        "cl_model_idx": conformer_idx.reshape(n, 1, 2).expand(n, S, 2),
        "cl_contrast_proc": contrast_proc,
        "cl_amp_free": amp_free,
        "cl_use_free": use_free,
        "cl_topology": rnd(n, S, N_ARCHETYPES),
        "cl_rg_A": _rng("rg_range_A", rnd(n, S)),
        "cl_aniso": (rnd(n, S, 2) - 0.5) * 2.0 * aw,
        "cl_hollowness": rnd(n, S),
        "cl_persistence": 0.70 + 0.29 * rnd(n, S),
        "cl_d_fractal": 1.5 + 1.2 * rnd(n, S),
        "cl_sheet_thick": 0.05 + 0.40 * rnd(n, S),
        "cl_quat": cl_quat,
        "cl_offset_A": torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=-1),
    }
    parts = [torch.as_tensor(fields[s.name]).reshape(n, -1) for s in THETA_LAYOUT]
    theta = torch.cat([q.to(torch.float64) for q in parts], dim=1).to(dtype)
    if return_diagnostics:
        return theta, {"cl_on_target": on_target_diag.to(dtype)}
    return theta


# ==============================================================================================
# PART 13 -- THE RENDERER: theta -> image  (paper: SI S3 Stages 1-5, Fig. 2)
# ==============================================================================================

def on_target(offset_A: torch.Tensor, cfg: RenderShape) -> torch.Tensor:
    """(B, S, 2) object positions -> (B, S): 1 where a clutter object sits on the molecule itself.

    For information only: no part of the image depends on it.
    """
    rr = offset_A.pow(2).sum(-1).sqrt()
    if cfg.r_inner_A is None:                 # cfg.with_bank(bank) was not called
        return torch.zeros_like(rr)
    return (rr < float(cfg.r_inner_A)).to(rr.dtype)


def to_v3_block(fields: dict[str, torch.Tensor], cfg: RenderShape,
                u: torch.Tensor | None = None) -> dict:
    """Collect everything the clutter step (render_v3_clutter) needs, from the unpacked theta.

    Also rebuilds each image's fixed random points `u` from its seed (derive_u), unless `u`
    is given.
    """
    t = fields["cl_t_pdb"]
    if u is None:
        u = derive_u(fields["seed"], cfg.n_slots, cfg.n_points_proc,
                     device=t.device)
    return dict(
        is_bank=t > 0,                       # the object contains part of the molecule
        model_idx=fields["cl_model_idx"].long(),
        win_start=fields["cl_win_start"],
        win_frac=t,                          # the fraction of atoms a piece keeps equals t
        t_pdb=t,
        contrast_proc=fields["cl_contrast_proc"],
        amp_free=fields["cl_amp_free"],
        use_free=fields["cl_use_free"],
        u=u,
        topology=fields["cl_topology"],
        rg_A=fields["cl_rg_A"],
        aniso=fields["cl_aniso"],
        hollowness=fields["cl_hollowness"],
        persistence=fields["cl_persistence"],
        d_fractal=fields["cl_d_fractal"],
        sheet_thickness=fields["cl_sheet_thick"],
        quat=fields["cl_quat"],
        offset_A=fields["cl_offset_A"],
        on_target=on_target(fields["cl_offset_A"], cfg),   # for information only
        present=fields["cl_present"],
        n_slots=cfg.n_slots,
        n_points_proc=cfg.n_points_proc,
        min_fragment_atoms=cfg.min_fragment_atoms,
    )


def render(
    theta: torch.Tensor,
    bank: torch.Tensor,
    cfg: RenderShape,
    *,
    rng_seed="theta",
) -> torch.Tensor:
    """Make the images: theta (B, 156) -> images (B, N, N). The five stages of the paper's SI S3.

    Inputs
        theta     (B, 156) image settings, from sample_theta (or edited by hand).
        bank      the conformer bank (M, R, 3, A).
        cfg       the RenderShape: box size, pixel size, where the noise spectrum comes from.
        rng_seed  how PyTorch's global random stream is seeded before drawing the noise fields
                  and the radii of shell-shaped clutter clouds:
                    "theta"    (default) from the FIRST image's seed, so the same theta batch
                               always gives the same images;
                    "ambient"  leave the stream as it is (you seed it yourself);
                    a number   seed it with that number.
                  The stream is shared by the whole batch, so one image's noise (like the shell
                  radii of its clutter) depends on the other images in its batch. Note that
                  seeding resets PyTorch's global random stream, which other code may use too.
    Output
        (B, N, N) images, each with mean 0 and standard deviation 1.
    """
    models = as_4d(bank)
    f = unpack_theta(theta, cfg.n_slots)
    B = theta.shape[0]
    dev = theta.device

    if rng_seed == "theta":
        torch.manual_seed(int(f["seed"].reshape(-1)[0].detach().item()) % (1 << 63))
    elif rng_seed is not None and rng_seed != "ambient":
        torch.manual_seed(int(rng_seed))

    num_pixels = torch.tensor(float(cfg.num_pixels), dtype=torch.float32, device=dev)
    pixel_size = torch.tensor(float(cfg.pixel_size), dtype=torch.float32, device=dev)

    index = f["conformer_idx"].long()
    sigma = f["sigma"].reshape(B, 1, 1)
    defocus = f["defocus"].reshape(B, 1, 1)
    b_factor = f["b_factor"].reshape(B, 1, 1)
    amp = f["amp"].reshape(B, 1, 1)
    snr = f["log10_snr"].reshape(B, 1, 1)

    # --- Stage 1: projection of the molecule (Eq. S14) ---------------------------------------
    signal = project_density(
        models[index[:, 0], index[:, 1]], f["quaternion"], sigma, f["shift"],
        num_pixels, pixel_size,
    )
    if cfg.noise_floor_std is not None:
        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # --- The noise reference (Eq. S15): blur the molecule ALONE and measure its strength. -----
    # This is a measurement, not a stage: it fixes the noise level, so that the SNR describes
    # the molecule and not the clutter added next.
    # The zero-padding is sized for a defocus of 3.5 micrometres, whatever the prior (Part 4).
    _pad = ctf_pad_px(int(signal.shape[-1]), float(cfg.pixel_size))
    main_image_post_ctf = apply_ctf(signal, defocus, b_factor, amp, pixel_size, pad_px=_pad)
    # get_snr creates the number 10 without saying on which device. On a GPU that would break
    # the backward pass, so the image's device is made the default device here.
    with torch.device(dev):
        noise_scale_main = get_snr(main_image_post_ctf, snr).reshape(-1, 1, 1)

    if cfg.noise_floor_std is not None:
        _main_std = main_image_post_ctf.flatten(1).std(dim=1).reshape(-1, 1, 1)
        _empty = _main_std < 1e-6
        if bool(_empty.any()):
            _ten = torch.tensor(10.0, device=signal.device, dtype=signal.dtype)
            _floor = float(cfg.noise_floor_std) / torch.sqrt(torch.pow(_ten, snr)).reshape(-1, 1, 1)
            noise_scale_main = torch.where(_empty, _floor, noise_scale_main)

    # --- Stage 2: clutter, added to the clean projection BEFORE the blur, -------------------
    # so the microscope blurs it exactly like the molecule. Images whose count is 0 get none.
    signal = render_v3_clutter(signal, models, to_v3_block(f, cfg), sigma, num_pixels, pixel_size)

    # --- Stage 3: the microscope blur (CTF, Eq. S16) of molecule plus clutter -----------------
    image = apply_ctf(signal, defocus, b_factor, amp, pixel_size, pad_px=_pad)

    if cfg.noise_floor_std is not None:
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Stage 4: noise (Eq. S10, S11, S17) ---------------------------------------------------
    # The shot-noise dose is lambda = clip(gamma * SNR, 1, 200) with the LINEAR SNR (Eq. S10).
    nk = dict(cfg.noise or {})
    snr_for_dose = torch.pow(torch.tensor(10.0, dtype=snr.dtype, device=dev), snr)
    noisy = add_noise(
        image, snr_for_dose, f["noise_w"],
        noise_scale=noise_scale_main,
        parameter_source=nk.get("parameter_source", "standard"),
        psd_exponent=DEFAULT_SYNTHETIC_PSD_EXPONENT,       # only for the last fallback spectrum
        poisson_dose_scale=POISSON_GAMMA,
        empirical_cache_file=nk.get("empirical_cache_file"),
        empirical_root_folder=nk.get("empirical_root_folder"),
        empirical_num_images=nk.get("empirical_num_images", 10000),
        empirical_selection_mode=nk.get("empirical_selection_mode", "random"),
        empirical_random_seed=nk.get("empirical_random_seed", 42),
        empirical_batch_size=nk.get("empirical_batch_size", 64),
    )

    # --- Stage 5: the edge band (Eq. S12), after the noise so the band copies noisy pixels, ---
    # then standardisation to mean 0 and standard deviation 1.
    return gaussian_normalize_image(_edge_band(noisy, f, cfg))


def _edge_band(img: torch.Tensor, f: dict, cfg: RenderShape) -> torch.Tensor:
    """Stage 5 for a batch: add the edge band to the images that have one (Eq. S12).

    The depth is band_width_frac x N pixels, and the side comes from theta. The softness of the
    band's edge (tau_b = 0.5 pixel) is fixed; it is not the same thing as side_tau, which only
    affects derivatives of the side choice.
    """
    depth_frac = f["band_width_frac"].reshape(-1).to(img.dtype)
    depth_frac = depth_frac * f["band_present"].reshape(-1).to(img.dtype)
    if not bool((depth_frac > 0).any()):
        return img
    n = img.shape[-1]
    return apply_edge_replication(
        img,
        depth_px=depth_frac * n,
        side=f["band_side"].reshape(-1).long(),
        side_onehot=f["band_side_onehot"],
    )


def render_from_phi(phi: torch.Tensor, n: int, seed: int, bank: torch.Tensor,
                    cfg: RenderShape, *, rng_seed="theta"):
    """The whole chain in one call: phi -> theta -> images. Returns (images, theta).

    Derivatives flow from the images back to phi.
    """
    theta = sample_theta(phi, n, seed, bank, cfg)
    images = render(theta, bank, cfg.with_bank(bank), rng_seed=rng_seed)
    return images, theta


# ==============================================================================================
# PART 14 -- THE DEFAULT PRIOR  (the paper's Table S3)
# ==============================================================================================
# Physical values, as printed in the paper's Table S3 (2 to 3 significant figures). Turn it into
# phi with pack_phi(DEFAULT_PRIOR), or copy the dictionary, change values, and pack the copy.
#
# Ranges are [lo, hi]; SNR is in log10; DEFOCUS in micrometres; SIGMA and rg_range_A in Angstrom;
# B_FACTOR in Angstrom^2; barcode_width_frac as a fraction of the box size. count_probs and
# side_weights are divided by their sum when packed.

DEFAULT_PRIOR = {
    "SIGMA": [0.261, 2.41], "DEFOCUS": [0.0551, 5.62], "B_FACTOR": [0.0871, 129.0], "SNR": [-5.8, -1.5],
    "MIX_RANGE_gaussian": [0.0, 0.784], "MIX_RANGE_fft": [0.0, 0.956], "MIX_RANGE_poisson": [0.0, 0.498],
    "CONC_fft": 8.49, "WHITE_FRAC_fft": 0.205, "REGIME_full_psd": 0.392, "REGIME_white": 0.151,
    "count_probs": [0.458, 0.153, 0.134, 0.118, 0.085, 0.051], "p_bank": 0.672, "p_duplicate": 0.391,
    "p_free_amp": 0.373, "rg_range_A": [1.2, 137.0], "contrast_range": [-0.92, 1.41],
    "amp_free_range": [0.0, 1.14], "aniso_log_halfwidth": 1.59, "radial_exponent": 1.26,
    "barcode_prob": 0.139, "barcode_width_frac": [0.0, 0.292], "side_weights": [0.239, 0.305, 0.238, 0.219],
    "depth_a": 0.913, "depth_b": 1.19,
}


__all__ = [
    # the three main calls
    "pack_phi", "sample_theta", "render", "render_from_phi",
    # set-up
    "RenderShape", "load_bank", "load_bank_pt", "pad_bank",
    # reading and writing theta and phi
    "unpack_phi", "pack_theta", "unpack_theta", "derive_u", "slot_index",
    "THETA_LAYOUT", "PHI_LAYOUT", "THETA_LEN", "PHI_LEN",
    # the default prior and a constant
    "DEFAULT_PRIOR", "POISSON_GAMMA",
    # measuring a noise spectrum from your own particles
    "estimate_empirical_noise_parameters",
]
