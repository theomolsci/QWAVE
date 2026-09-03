"""VAE-based Slater-determinant generator for SqDRIFT / SQD workflows."""

import numpy as np
import torch
import torch.nn as nn


class _VAE(nn.Module):
    def __init__(self, n_visible, latent_dim):
        super().__init__()
        h = max(256, 8 * n_visible)

        self.encoder_net = nn.Sequential(
            nn.Linear(n_visible, h),     nn.ReLU(),
            nn.Linear(h, h // 2),        nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(h // 2, latent_dim)
        self.fc_logvar = nn.Linear(h // 2, latent_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, h // 2), nn.ReLU(),
            nn.Linear(h // 2, h),          nn.ReLU(),
            nn.Linear(h, n_visible),        nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def _fix_particle_number_batch(v_batch, p_batch, target_Ne):
    """Greedy vectorized particle-number correction for a batch of binary samples."""
    v_out  = v_batch.copy()
    counts = v_out.sum(axis=1)

    over_mask = counts > target_Ne
    if over_mask.any():
        vo   = v_out[over_mask]
        po   = p_batch[over_mask]

        sort_idx = np.argsort(po, axis=1)

        excess = counts[over_mask] - target_Ne

        ones_sorted = np.take_along_axis(vo, sort_idx, axis=1)
        cum_ones    = np.cumsum(ones_sorted, axis=1)

        flip_sorted = (ones_sorted == 1) & (cum_ones <= excess[:, np.newaxis])

        inv_idx  = np.argsort(sort_idx, axis=1)
        flip_orig = np.take_along_axis(flip_sorted, inv_idx, axis=1)

        vo[flip_orig] = 0
        v_out[over_mask] = vo

    under_mask = counts < target_Ne
    if under_mask.any():
        vu   = v_out[under_mask]
        pu   = p_batch[under_mask]

        sort_idx = np.argsort(-pu, axis=1)

        deficit = target_Ne - counts[under_mask]

        zeros_sorted = (np.take_along_axis(vu, sort_idx, axis=1) == 0).astype(np.int8)
        cum_zeros    = np.cumsum(zeros_sorted, axis=1)

        flip_sorted  = (zeros_sorted == 1) & (cum_zeros <= deficit[:, np.newaxis])

        inv_idx   = np.argsort(sort_idx, axis=1)
        flip_orig = np.take_along_axis(flip_sorted, inv_idx, axis=1)

        vu[flip_orig] = 1
        v_out[under_mask] = vu

    return v_out


def _fix_spin_resolved_batch(v_batch, p_batch, n_alpha, n_beta, n_spatial):
    """Enforce spin-sector particle numbers independently for a full batch."""
    v_out = v_batch.copy()
    v_out[:, :n_spatial] = _fix_particle_number_batch(
        v_batch[:, :n_spatial], p_batch[:, :n_spatial], n_beta)
    v_out[:, n_spatial:] = _fix_particle_number_batch(
        v_batch[:, n_spatial:], p_batch[:, n_spatial:], n_alpha)
    return v_out


def _decode_in_chunks(model, z_tensor, chunk_size=8192):
    """Decode latent vectors in chunks; returns a float32 CPU numpy array."""
    parts = []
    with torch.no_grad():
        for start in range(0, z_tensor.shape[0], chunk_size):
            chunk = z_tensor[start : start + chunk_size]
            parts.append(model.decode(chunk).cpu().numpy())
    return np.concatenate(parts, axis=0)


def generate_new_determinants(
    sampled_bitstrings,
    existing_pool,
    total_electrons,
    n_qubits,
    n_alpha=None,
    n_beta=None,
    num_spatial_orbitals=None,
    latent_dim=180,
    n_epochs=1000,
    n_finetune_epochs=300,
    batch_size=1024,
    lr=1.5e-3,
    finetune_lr=7.5e-4,
    beta_max=0.3,
    n_prior_samples=500000,
    perturb_scale=6.0,
    init_state_dict=None,
):
    """Train a VAE on sampled_bitstrings and return newly discovered determinants."""
    spin_resolved = (
        n_alpha is not None
        and n_beta is not None
        and num_spatial_orbitals is not None
    )

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    else:
        device = torch.device('cpu')
    print(f"  [device] {device}")

    X = torch.tensor(sampled_bitstrings, dtype=torch.float32).to(device)
    effective_batch = min(batch_size, len(sampled_bitstrings))

    def _epoch_batches():
        perm = torch.randperm(len(X), device=device)
        for start in range(0, len(X), effective_batch):
            yield X[perm[start : start + effective_batch]]

    model = _VAE(n_qubits, latent_dim).to(device)

    _torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    _compile_ok = _torch_version >= (2, 4) if device.type == 'mps' else _torch_version >= (2, 0)
    if _compile_ok:
        try:
            _compiled = torch.compile(model)
            with torch.no_grad():
                _compiled(torch.zeros(2, n_qubits, device=device))
            model = _compiled
        except Exception:
            pass

    if init_state_dict is not None:
        _m_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        _m_load.load_state_dict(init_state_dict)
        active_epochs = n_finetune_epochs
        active_lr     = finetune_lr
        mode_label    = "fine-tuning"
    else:
        active_epochs = n_epochs
        active_lr     = lr
        mode_label    = "training"

    optimizer = torch.optim.Adam(model.parameters(), lr=active_lr)

    print(f"\n{mode_label.capitalize()} VAE  [n_qubits={n_qubits}  "
          f"latent_dim={latent_dim}  epochs={active_epochs}  "
          f"training_samples={len(X)}]")

    for epoch in range(active_epochs):
        beta = beta_max * min(1.0, 2.0 * epoch / active_epochs)
        model.train()
        epoch_loss = torch.zeros((), device=device)
        epoch_bce  = torch.zeros((), device=device)
        epoch_kl   = torch.zeros((), device=device)
        for batch in _epoch_batches():
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            bce  = nn.functional.binary_cross_entropy(recon, batch, reduction='sum')
            kl   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + beta * kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            epoch_bce  += bce.detach()
            epoch_kl   += kl.detach()
        if (epoch + 1) % 100 == 0:
            n = len(X)
            print(f"  epoch {epoch + 1:>5}/{active_epochs}  "
                  f"loss={epoch_loss.item()/n:.4f}  "
                  f"bce={epoch_bce.item()/n:.4f}  "
                  f"kl={epoch_kl.item()/n:.4f}  "
                  f"beta={beta:.3f}")

    model.eval()

    def _fix_batch(v_batch, p_batch):
        """Apply particle-number correction to the entire batch at once."""
        if spin_resolved:
            return _fix_spin_resolved_batch(
                v_batch, p_batch, n_alpha, n_beta, num_spatial_orbitals)
        return _fix_particle_number_batch(v_batch, p_batch, total_electrons)

    all_batches = []

    z_prior    = torch.randn(n_prior_samples, latent_dim, device=device)
    probs_prior = _decode_in_chunks(model, z_prior, chunk_size=8192)
    del z_prior

    rnd_prior   = np.random.rand(n_prior_samples, n_qubits)
    v_prior     = (rnd_prior < probs_prior).astype(np.int8)
    v_prior_fix = _fix_batch(v_prior, probs_prior).astype(np.int8)
    all_batches.append(v_prior_fix)

    perturb_chunk = 4096
    X_np = sampled_bitstrings

    for start in range(0, len(X_np), perturb_chunk):
        end   = min(start + perturb_chunk, len(X_np))
        batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)

        with torch.no_grad():
            mu_t, logvar_t = model.encode(batch)
            std_t          = torch.exp(0.5 * logvar_t)
            eps_t          = torch.randn_like(std_t)
            z_perturbed    = mu_t + perturb_scale * std_t * eps_t
            probs_np       = model.decode(z_perturbed).cpu().numpy()

        chunk_size_actual = end - start
        rnd_chunk   = np.random.rand(chunk_size_actual, n_qubits)
        v_chunk     = (rnd_chunk < probs_np).astype(np.int8)
        v_chunk_fix = _fix_batch(v_chunk, probs_np).astype(np.int8)
        all_batches.append(v_chunk_fix)

    all_samples = np.concatenate(all_batches, axis=0)
    unique_samples = np.unique(all_samples, axis=0)

    if spin_resolved:
        beta_counts  = unique_samples[:, :num_spatial_orbitals].sum(axis=1)
        alpha_counts = unique_samples[:, num_spatial_orbitals:].sum(axis=1)
        valid_mask   = (beta_counts == n_beta) & (alpha_counts == n_alpha)
    else:
        total_counts = unique_samples.sum(axis=1)
        valid_mask   = total_counts == total_electrons

    valid_samples = unique_samples[valid_mask]

    _chars = valid_samples.astype(np.uint8) + np.uint8(ord('0'))
    ranked = [row.tobytes().decode('ascii') for row in _chars]
    new_determinants = [s for s in ranked if s not in existing_pool]

    spin_label = "spin-resolved" if spin_resolved else "total-Ne"
    print(f"Generated {len(unique_samples)} unique determinants  "
          f"-> {len(valid_samples)} valid ({spin_label})  "
          f"-> {len(new_determinants)} new")

    _m = model._orig_mod if hasattr(model, '_orig_mod') else model
    state_dict_cpu = {k: v.cpu() for k, v in _m.state_dict().items()}
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return new_determinants, state_dict_cpu
