# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: 1.3
#   kernelspec:
#     display_name: diffusion-project
#     language: python
#     name: python3
#   language_info:
#     name: python
#     version: 3.11.0
# ---

# %%
# Cell 1
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from diffusion_sim.io.npz import load_npz
from diffusion_sim.io import load_h5
from diffusion_sim.plotting import embed_mp4, plot_energy, plot_std, plot_density_vs_radius, plot_scaled_density_vs_radius, plot_msd_by_charge
from scripts.compare_solvers import animate_comparison_mp4, plot_divergence_sweep

# Set the limit to 50 MB (default is ~20 MB)
plt.rcParams['animation.embed_limit'] = 50.0


def show_existing_mp4(path, width=600):
    return embed_mp4(path, width=width, embed=False)

# %% [markdown]
# ### 700 particles different init - 650 q=1, 50 q=10, D = 1.0, seed = 5

# %%
    
sim = load_h5("data\\20260211\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D1.0.h5")
print(sim["meta"])
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260212\\rk23_N300_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n270-30_save0.001_diff_D1.0.mp4")

# %% [markdown]
# ### 980 particles - 910 q=1, 70 q=10, D = 1.0, seed = 1

# %%
sim = load_h5("data\\20260210\\rk23_N980_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n910-70_save0.001_diff_D1.0.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260215\\rk23_N980_t100.0_k3.0_rtol0.001_atol0.001_q1.0-25.5_n655-45_save_4e-4_diff_D1.0.mp4")

# %% [markdown]
# ### 700 particles - 650 q=1, 50 q=10, D = 0.1

# %%
sim = load_h5("data\\20260211\\rk23_N700_t50_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D0.1.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260215\\rk23_N700_t50_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save_4e-4_diff_0.1.mp4")

# %% [markdown]
# ### 700 particles - 650 q=1, 50 q=10, D = 0

# %%
sim = load_h5("data\\20260211\\rk23_N700_t50_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260215\\rk23_N700_t50_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save_1e-3.mp4")

# %%
sim = load_h5("data\\20260211\\rk23_N700_t50_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001.h5")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_density_vs_radius(sim, times=[5,7,25, 30, 50], charge_value=10.0, ax=ax1, show=False, scaled=True)
plot_density_vs_radius(sim, times=[5,7,25, 30, 50], charge_value=1.0, ax=ax2, show=False, scaled=True)
ax1.set_title("Density vs Radius (Charge = 10.0)")
ax2.set_title("Density vs Radius (Charge = 1.0)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 700 particles - 650 q=1, 50 q=10, D = 1.0 init radius = 0.3

# %%
sim = load_h5("data\\20260222\\rk23_N700_t100.0_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D1.0.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260222\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save_1e-3_init_r0.3.mp4")
