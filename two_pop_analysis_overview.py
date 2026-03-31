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
# ### 100 particles 40 q=2, 60 q=1, no diffusion

# %%
    
sim = load_h5("data\\20260206\\rk23_N100_t0.1_k3.0_rtol0.001_atol0.001_q1.0-2.0_n60-40_save0.001.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260206\\rk23_N100_t0.1_k3.0_rtol0.001_atol0.001_q1.0-2.0_n60-40_save0.001.mp4")
    

# %% [markdown]
# # 350 particles: 210 q=1, 140 q=5, No diffusion

# %%
    
sim = load_h5("data\\20260206\\rk23_N350_t1_k3.0_rtol0.001_atol0.001_q1.0-5.0_n210-140_save0.001.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260206\\rk23_N350_t1_k3.0_rtol0.001_atol0.001_q1.0-5.0_n210-140_save0.001.mp4")
    

# %%
    
sim = load_h5("data\\20260206\\rk23_N350_t1_k3.0_rtol0.001_atol0.001_q1.0-5.0_n210-140_save0.001_diff_D3.0.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260206\\rk23_N350_t1_k3.0_rtol0.001_atol0.001_q1.0-5.0_n210-140_save0.001_diff_D3.0.mp4")
    

# %%
    
sim = load_h5("data\\20260206\\rk23_N350_t5_k3.0_rtol0.001_atol0.001_q1.0-5.0_n300-50_save0.001_diff_D3.0.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260206\\rk23_N350_t5_k3.0_rtol0.001_atol0.001_q1.0-5.0_n300-50_save0.001_diff_D3.0.mp4")
    

# %% [markdown]
# ### 700 particles - 600 q=1, 100 q=5, D = 1.0

# %%
    
sim = load_h5("data\\20260208\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-5.0_n600-100_save0.001_diff_D1.0.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260208\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-5.0_n600-100_save0.001_diff_D1.0.mp4")

# %% [markdown]
# ### 700 particles - 650 q=1, 50 q=10, D = 1.0

# %%
    
sim = load_h5("data\\20260209\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D1.0.h5")
plot_energy(sim)
plot_std(sim, show_theory=True)
show_existing_mp4("videos\\20260209\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D1.0.mp4")

# %%
sim = load_h5("data\\20260209\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D1.0_with_density_by_charge.h5")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_density_vs_radius(sim, times=[25, 50, 75], charge_value=10.0, ax=ax1, show=False)
plot_density_vs_radius(sim, times=[25, 50, 75], charge_value=1.0, ax=ax2, show=False)
ax1.set_title("Density vs Radius (Charge = 10.0)")
ax2.set_title("Density vs Radius (Charge = 1.0)")
fig.tight_layout()
plt.show()

# %%
sim = load_h5("data\\20260209\\rk23_N700_t100_k3.0_rtol0.001_atol0.001_q1.0-10.0_n650-50_save0.001_diff_D1.0_with_density_by_charge.h5")
plot_msd_by_charge(                                                                                                                                           
    sim,                                                                                                                                                      
    particle_indices_by_charge={1.0: 12, 10.0: 670},                                                                                                          
    origin_window=0.05,
    same_plot=True,                                                                                                                                        
)
