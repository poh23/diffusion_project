### A) Scientific Purpose & Scope                                                                                                                             
                                                                                                                                                                
  1. What real system is being simulated (physics, chemistry, biology, etc.)?                                                                                
  2. What is the primary research question this code should answer?                                                                                             
  3. What are the governing equations (please write them explicitly)?                                                                                           
  4. Which variables are state variables, and which are derived observables?                                                                                    
  5. What core assumptions are in force (for example overdamped, conservative, dilute, mean-field)?                                                             
  6. Are the equations deterministic, stochastic, or mixed?                                                                                                     
  7. What unit system is used (SI, cgs, reduced, fully dimensionless)?                                                                                          
  8. What parameter ranges are scientifically meaningful in this project?                                                                                       
  9. What system sizes are expected (typical N, small test N, large production N)?                                                                              
  10. Which quantities should be conserved, and which are allowed to drift?   

  A) 1. its a physics system - we use physical asumptions
  2. we want to understand the behavior of 2 particle populations with forces and diffusion between. now we"re trying with a power law and each population hhas a different " charge" but we next want to implement different repulsive forces like exponential and so on.
  3. we are in an overdamped system, so v = mu * F, and v_i = sum_j coupling * (q_i * q_j) * r_ij / (|r_ij|^(k+2)). In addition in order to simulate diffusion/ random walk, to every time step we add a random step dx = np.sqrt(2.0 * diffusion_coeff * dt) * noise where noise is a random variable with a gaussian pdf and set variance.
  5. the force is overdamped
  6. the equations are mixed since in each time step there is the deterministic step and a random step
  7. the unit system is dimensionless currently
  9. typical N is 700 - 1000, the small test N is 300, and large production is 1000-3000
  10. since we are in an overdamped system there is no conservation of energy, however i look at the energy of the system vs. time in oreder to check that its decaying and they"re no spikes.

                                                                                                                                                                
  ### B) Interaction / Force Model                                                                                                                              
                                                                                                                                                                
  1. What interaction law(s) are implemented (exact formulas)?                                                                                                  
  2. Is there a potential energy function associated with each force term?                                                                                      
  3. Are forces pairwise additive, many-body, or both?                                                                                                          
  4. How are short-range singularities handled (softening, regularization, minimum distance)?                                                                   
  5. Are there cutoffs, and if yes are they hard or smoothly switched?                                                                                          
  6. If cutoffs exist, is there a potential/force shifting scheme at cutoff?                                                                                    
  7. Are interactions reciprocal (Newton's 3rd law), or can they be non-reciprocal?                                                                             
  8. Are there particle types/species with different cross-interactions?
  9. Are external fields present (constant, spatially varying, time-dependent)?                                                                                 
  10. Are there geometric constraints (walls, traps, manifolds, fixed particles)?                                                                               
  11. What boundary conditions are used (periodic, reflective, absorbing, mixed)?                                                                               
  12. Are stochastic force terms included (white/colored noise), and how exactly are they applied?

  B)
  1. v_i = sum_j coupling * (q_i * q_j) * r_ij / (|r_ij|^(k+2)) currently
  2. if k == 0.0:
        pe_i = -coupling * (q_i * q_j) * np.log(|r_ij)
    else:
        pe_i = coupling * (q_i * q_j) / (k * |r_ij|^k)
  3. forces are pairwise additive
  4. the is a minimum distanc r_floor - r_ij_eff = r_ij if r_ij > r_floor else r_floor 
  5. there are no explicit cutoffs
  7. currently theyre reciprocal
  8. currently there a 2 types of particles with different charges. we may want to implement more than 2 populations and implenet 2 types of repulsive forces
  9. currently there arent any external fields
  10. currently no geomrtric constraints except minimal distance.
  11. no boundary conditions currently
  12. additional to the deterministic movement as a result of the forces, the particles have brownian motion meaning each particle conucts a random walk like - dx = np.sqrt(2.0 * diffusion_coeff * dt) * noise where noise is a random variable with a gaussian pdf and set variance.                                          
                                                                                                                                                                
  ### C) Numerical Update / Integration                                                                                                                         
                                                                                                                                                                
  1. Which integrators are currently implemented?                                                                                                               
  2. Which integrator is the default and why?                                                                                                                   
  3. For each integrator, what state variables are advanced each step?                                                                                          
  4. Is timestep fixed or adaptive by default?                                                                                                                  
  5. If adaptive, what error norm and controller are used?                                                                                                      
  6. What are default dt, rtol, atol, max_step, and stopping criteria?                                                                                          
  7. Are there event conditions (blow-up, collision, escape, steady state) that terminate runs?                                                                 
  8. What known stability limits should users respect?                                                                                                          
  9. What known failure modes should users watch for?                                                                                                           
  10. How is randomness seeded (global seed, per-run seed, per-component seed)?                                                                                 
  11. What level of reproducibility is expected across runs and across machines?                                                                                
  12. What diagnostics exist for integration quality (energy drift, invariant violation, residuals)?      
c)
  1. scipy's rk23, rk45, dop853, and my implementation of fix step rk2, rk4 but the fixed step integrator are deprecated.
  2. the default is rk23 since with it i implemented a random step after each rk23 step. i use rk23 instead of rk45 becausethe random step makes the error of the integration more irrelevant
  3. the positions are advanced in each step.
  4. in rk23, rk45 and dop852 which are the main untegrators we use the timesteps are adaptive
  5. I use a the builtin rk23 steps of scipy.
  6. there is no default dt beacuse we use adaptive timesteps, there is no default maxstep, default rtol and atol are -     rtol=1e-6,
    atol=1e-6
  7. no
  8. there is no way to know before the simulation the stability, or at least i dont know how to know
  9. unfortunatly when the simulation is running currently there is know way to know that it blows up only after the simulation completed and we look at the energy/ positions.
  10. there is a random seed for the initial positions of the particles, and there is a random seed for the random steps.
  11. i want it to be reproducable
  12. we look for spikes in the energy vs time, and the standard deviation of the particles positions is supposed to behave without diffusion as  t^(1/(k+2)) and with diffusion when the particles are far from each other as t^(1/2)                                             
                                                                                                                                                                
  ### D) Data Structures & I/O                                                                                                                                  
                                                                                                                                                                
  1. What is the exact state layout (positions, velocities, masses, charges, types, etc.)?                                                                      
  2. What are expected array shapes and dtypes for core tensors?                                                                                                
  3. What file formats are used for outputs (csv, npz, hdf5, etc.)?                                                                                             
  4. What is the output cadence (every step, every k steps, event-based)?                                                                                       
  5. Where are outputs written by default?                                                                                                                      
  6. What naming conventions are used for run folders/files?                                                                                                    
  7. What metadata is recorded (parameters, seed, code version, timestamp)?                                                                                     
  8. What plotting/animation scripts exist, and what dependencies do they require?                                                                              

D)
1.   positions=np.empty((0, config.n_particles, 2), dtype=np.float64),
times=np.empty(0, dtype=np.float64),
energy=np.empty(0, dtype=np.float64),
std=np.empty(0, dtype=np.float64),
final_positions=r_final,
charges=charges,
density=None,
radii=None,
meta=meta,
saved_path=str(out_path)
  2. 
  3. mainly hfd5 but also npz,.
  4. according to save_every parameter configured for each simulation 
  5+6. default: `data/YYYYMMDD/<method>_N<N>_t<T>_k<k>_rtol<rtol>_q<charge1>_<charge2>_n<num_charge1>_<num_charge2>_atol<atol>_save<save_every>_Diff<Diffusion_const>.hdf5`
  7. example metadata:   n_particles=config.n_particles,
                k=config.k,
                v0=config.v0,
                l=config.l,
                t_duration=config.t_duration,
                save_every=config.save_every,
                method=config.method,
                seed=config.seed,
                r_floor=config.r_floor,
                init_radius=config.init_radius,
                charge_values=config.charge_values,
                charge_counts=config.charge_counts,
                t0=config.t0,
                rtol=config.rtol,
                atol=config.atol,
                first_step=config.first_step if config.method == "rk23" else None,
                max_step_global=config.max_step_global if config.method == "rk23" else None,
                eta=config.eta if config.method == "rk23" else None,
                recompute_every=config.recompute_every if config.method == "rk23" else None,
                diffusion=config.diffusion if config.method == "rk23" else None,
                diffusion_coeff=config.diffusion_coeff if config.method == "rk23" else None,
                diffusion_seed=config.diffusion_seed if config.method == "rk23" else None,
                diffusion_noise_var=config.diffusion_noise_var if config.method == "rk23" else None,
                rk23_stats=rk23_stats if config.method == "rk23" else None,
                batch_every=config.batch_every,
                target_batch_mb=config.target_batch_mb,
                max_wall_time=config.max_wall_time,
                resume_from=str(out_path) if resumed_flag else None,
                resumed=resumed_flag,
                completed=completed,
                out_format=config.out_format,
                elapsed_sec=elapsed_sec,
  8. currently i use only plotting.py script its dependencies are from scipy.ndimage import gaussian_filter1d
 matplotlib.pyplot imageio_ffmpeg from matplotlib.animation import FuncAnimation FFMpegWriter    

  ### E) Repo Structure & Entry Points                                                                                                                          
                                                                                                                                                                
  1. What is the primary entry point for running simulations?                                                                                                   
  2. Is execution intended via script, module (python -m), or CLI tool?                                                                                         
  3. Where is config parsing implemented?                                                                                                                       
  4. Which modules are most critical for model physics?                                                                                                         
  5. Which modules are most critical for numerics/integration?                                                                                                  
  6. Are there tests, and where are they located?                                                                                                               
  7. Are there notebooks intended as official workflows or just exploratory?                                                                                    
  8. What are the top gotchas new contributors hit first?   

E)
  1. diffusion_sim.cli
  2. cli tool - diffusion_sim.cli
  3. in cli.py
  4. none, i dont use
  5. the scipy integrators, and voronoi area calculattion module
  6. there are some test but not enough in the tests directory
  7. i use the notebooks to display graphs of the simulation runs
  8. there wont be new contributors
                                                                                                    
                                                                                                                                                                
  ### F) Running The Project (Very Important)                                                                                                                   
                                                                                                                                                                
  1. Is uv the recommended workflow in this repo?                                                                                                               
  2. If yes, what is the exact required uv install method for your team?                                                                                        
  3. If not using uv, what is the supported fallback workflow?                                                                                                  
  4. What Python versions are officially supported?                                                                                                             
  5. What is the exact command to create the environment?                                                                                                       
  6. What is the exact command to install dependencies?                                                                                                         
  7. Are there optional dependency groups (dev, plotting, docs, test)?                                                                                          
  8. What is the exact minimal demo command that should work on a fresh setup?                                                                                  
  9. What command runs a standard full simulation?                                                                                                              
  10. What command runs tests?                                                                                                                                  
  11. What command reproduces a key figure/result from your work?                                                                                               
  12. Are there required environment variables (and defaults)?                                                                                                  
  13. Are there OS-specific instructions for Windows vs Linux vs macOS?                                                                                         
  14. Are external datasets/files required before running?                                                                                                      
  15. Where should users fetch those files, and where should they place them?                                                                                  
  16. What are expected runtime and memory for minimal, typical, and large runs?    

F)
  1. yes, uv is recommended
  2. just regular installment of uv?
  3. uv will always be used
  4. requires-python = ">=3.11"
  5+6. uv does it automatically
  7. currently there is a dev dependency woth the notebook dependency only
  9. diffusion.cli (configurations)
  10. dont know
  11. currently there is no command, i import the simulation data to a jupyter notebook cell and plot it using the plotting functions
  12. no
  13. no
  14. it could be helpful having a json configuration file for running the simulation
  15. there are saved eaxamples in the main diractory, there is no marked place or directory to save htem
  16. large runs run for about several hours.
                                                                        
                                                                                                                                                                
  ### G) Definition Of Correct                                                                                                                                  
                                                                                                                                                                
  1. What concrete outputs indicate a run is correct?                                                                                                           
  2. What baselines/reference results are considered canonical?                                                                                                 
  3. Which quantitative metrics are used for validation?                                                                                                        
  4. What tolerances are acceptable for those metrics?                                                                                                          
  5. Are visual checks (plots/trajectories) part of acceptance criteria?                                                                                        
  6. What should fail fast versus warn only? 

G)
  1. i look at the energy vs time and see if it decays with no spikes. its only a sanity check i dont know what is truly correct when running with 2 populations
  2. with one population and no diffusion there is a self similar solution.
  3. energy and standard deviation, in the future ill add density vs. radius
  4. it depends, the main atol, rtol and eta of rk23 was chosen by finding the fastest run time, and such that each position of a particle is with 10^-3 error of the position of it integrated using dop853.
  5. the energy and std plots

                                                                                                                                                                
  ### H) Contribution & Style                                                                                                                                   
                                                                                                                                                                
  1. What formatting/linting tools are required (black, ruff, etc.)?                                                                                            
  2. Are type hints mandatory in new/modified code?                                                                                                             
  3. What docstring style should contributors follow?                                                                                                           
  4. What branch and commit message conventions should be used?                                                                                                 
  5. What is the preferred pattern for adding a new force law?                                                                                                  
  6. What is the preferred pattern for adding a new observable/diagnostic?

H)
  1-6. there are no contribution & style prefrences, you can choose them.                                                                                     
                                                                                                                                                                
  If any answer is unknown, please write: I don't know yet - please specify assumptions.