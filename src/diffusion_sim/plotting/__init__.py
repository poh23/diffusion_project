"""Plotting package organized by analysis area."""

from .animation_video import animate_mp4, embed_mp4, save_mp4
from .density_profile_plots import (
    plot_density_vs_radius,
    plot_inverse_density_squared_vs_radius_squared,
    plot_scaled_density_vs_radius,
)
from .energy_metrics import plot_energy, plot_energy_by_charge
from .mean_radial_separation_plots import (
    compute_signed_mean_radius_difference,
    plot_scaled_signed_mean_radius_difference_vs_ratio_by_k,
    plot_signed_mean_radius_difference,
    plot_signed_mean_radius_difference_vs_charge_value_ratio,
    plot_signed_mean_radius_difference_vs_ratio,
)
from .mixing_inner_core_structure import (
    compute_mixing_metric,
    plot_inner_particles_selfsimilar_radius,
    plot_mixing_metric,
)
from .radial_comparison import (
    plot_high_inner_radius_vs_charge_product,
    plot_inner_radius_ratio_power_vs_charge_value_ratio,
    plot_low_inner_radius_vs_charge_product,
    plot_low_inner_radius_vs_low_population,
    plot_low_radius_vs_high_population_by_initial_radii,
    plot_low_radius_vs_high_population_by_low_population,
    plot_radial_force_balance,
    plot_special_radius_ratio_vs_charge_population_ratio,
)
from .std_diagnostics import compute_std_by_charge, plot_msd_by_charge, plot_std, plot_std_by_charge
from .wasserstein_radial_separation_plots import (
    compute_signed_radial_wasserstein,
    plot_scaled_signed_radial_wasserstein_vs_ratio_by_k,
    plot_signed_radial_wasserstein,
    plot_signed_radial_wasserstein_vs_ratio,
)

__all__ = [
    "animate_mp4",
    "compute_mixing_metric",
    "compute_std_by_charge",
    "compute_signed_mean_radius_difference",
    "compute_signed_radial_wasserstein",
    "embed_mp4",
    "plot_density_vs_radius",
    "plot_energy",
    "plot_energy_by_charge",
    "plot_high_inner_radius_vs_charge_product",
    "plot_inner_particles_selfsimilar_radius",
    "plot_inner_radius_ratio_power_vs_charge_value_ratio",
    "plot_inverse_density_squared_vs_radius_squared",
    "plot_low_inner_radius_vs_charge_product",
    "plot_low_inner_radius_vs_low_population",
    "plot_low_radius_vs_high_population_by_initial_radii",
    "plot_low_radius_vs_high_population_by_low_population",
    "plot_mixing_metric",
    "plot_msd_by_charge",
    "plot_radial_force_balance",
    "plot_scaled_density_vs_radius",
    "plot_scaled_signed_mean_radius_difference_vs_ratio_by_k",
    "plot_scaled_signed_radial_wasserstein_vs_ratio_by_k",
    "plot_signed_mean_radius_difference",
    "plot_signed_mean_radius_difference_vs_charge_value_ratio",
    "plot_signed_mean_radius_difference_vs_ratio",
    "plot_signed_radial_wasserstein",
    "plot_signed_radial_wasserstein_vs_ratio",
    "plot_special_radius_ratio_vs_charge_population_ratio",
    "plot_std",
    "plot_std_by_charge",
    "save_mp4",
]
