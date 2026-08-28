## Data

All data is included directly in this repository. First clone the repository:

```
git clone https://github.com/Rares-Andrei/distributed_working_memory.git
cd distributed_working_memory
```

## Environment

To run the scripts and notebooks, install the environment first and activate:
```
conda env create -f environment.yml
conda activate WM_human
```

## Repository Overview

The project is organized into functional modules.  
Below is the directory structure (folders only), followed by a description of what each folder contains.

```text
├── data
│   ├── processed
│   │   └── macaque
│   │   └── (files for human model)
│   └── raw
│       ├── Centroids_schaefer100
│       ├── Centroids_schaefer200
│       ├── Glasser_map
│       ├── PET_neurotransmitters
│       ├── brain_annotations
│       │   ├── Glasser
│       │   └── Schaefer
│       ├── hansen_data
│       ├── macaque
│       ├── structural_connectivity_schaefer100_EBRAINS
│       └── structural_connectivity_schaefer200_EBRAINS
├── data_analysis_autorad_hierarchy
│   ├── notebooks
│   └── results
│       ├── autoradiography_nmda_ampa_schaefer100
│       ├── long_rage_projections_strengths_schaefer100
│       └── t1t2_and_hierarchy_schaefer100
├── Figures
│   ├── revision
│   │   └── plots
│   │       ├── FC_grad
│   │       ├── delays
│   │       ├── main_task_inh
│   │       ├── vary_Z
│   │       └── wc
│   └── saved_plots
│       ├── Figure_1
│       ├── Figure_2
│       ├── Figure_3_CE
│       ├── Figure_3_DF
│       ├── Figure_4
│       ├── Figure_5
│       ├── Figure_6B
│       ├── Figure_6D
│       ├── Figure_6E
│       ├── Supp_Figure_2
│       ├── Supp_Figure_3
│       └── Supp_Figure_5
├── models
│   ├── macaque
│   │   ├── config
│   │   ├── notebooks
│   │   ├── results
│   │   └── src_utils
│   ├── model_autoradiography_nmda_ampa
│   │   ├── config
│   │   ├── notebooks
│   │   ├── results
│   │   │   ├── connectome
│   │   │   ├── firing_rates_from_notebook_sim
│   │   │   └── main_task
│   │   ├── scripts
│   │   └── src
│   ├── model_pet_nmda
│   │   ├── config
│   │   ├── notebooks
│   │   │   └── rev
│   │   ├── results
│   │   │   ├── combined_panels
│   │   │   ├── connectome
│   │   │   ├── distractor_resistance
│   │   │   ├── g_alpha_flat_nmda_rev
│   │   │   ├── g_alpha_flat_recurrent_e_recurrent_hierarchy_trial3
│   │   │   ├── g_alpha_recurrent_e_recurrent_hierarchy
│   │   │   ├── g_alpha_sln04_recurrent_hierarchy
│   │   │   ├── g_alpha_sln04_recurrent_hierarchy_trial2
│   │   │   ├── g_alpha_sln05_recurrent_hierarchy
│   │   │   ├── g_alpha_sln05_rev
│   │   │   ├── main_task
│   │   │   ├── main_task_global_coupling_zero
│   │   │   ├── main_task_inh
│   │   │   ├── over_under_active
│   │   │   └── phase_diagram_G_alpha
│   │   │       ├── g_alpha_random_seeds_per_G_alpha
│   │   │       └── g_alpha_random_seeds_per_G_alpha_corrected_spont_fun
│   │   ├── scripts
|   |   ├── scripts_flattening
│   │   └── src
│   ├── model_pet_nmda_FC_grad
│   │   ├── config
│   │   ├── notebooks
│   │   ├── results
│   │   │   └── main_task
│   │   ├── scripts
│   │   └── src
│   ├── model_pet_nmda_WC
│   │   ├── config
│   │   ├── notebooks
│   │   │   └── rev
│   │   ├── results
│   │   │   └── main_task_wc
│   │   ├── scripts
│   │   └── src
│   ├── model_pet_nmda_delays
│   │   ├── config
│   │   ├── notebooks
│   │   │   └── rev
│   │   ├── results
│   │   │   └── main_task
│   │   ├── scripts
│   │   └── src
│   ├── model_pet_nmda_inferred-ampa
│   │   ├── config
│   │   ├── results
│   │   ├── scripts
│   │   └── src
│   ├── model_pet_nmda_mixed_selectivity
│   │   ├── config
│   │   ├── notebooks
│   │   ├── results
│   │   │   ├── G_selectivity_param_sweep
│   │   │   ├── decoding
│   │   │   │   └── figs
│   │   │   └── firing_rates_from_notebook_sim
│   │   ├── scripts
│   │   └── src
│   └── model_pet_nmda_vary_Z
│       ├── config
│       ├── results
│       ├── scripts
│       └── src
├── notebooks
│   └── results
│       ├── literature_map_glasser_schaefer
│       └── macaque
└── scripts
    ├── compile_phase_diagram_results
    └── data_processing
```
### data/

Stores all raw and processed datasets used by the models.

#### `data/raw/` - Original external datasets

- **`Centroids_schaefer100`**, **`Centroids_schaefer200`** - Parcellation centroids  
- **`Glasser_map`** - Glasser atlas information  
- **`PET_neurotransmitters`** - Neurotransmitter PET NMDA map
- **`brain_annotations/Glasser`**, **`brain_annotations/Schaefer`** - Brain regions data for plotting and analysis based on Glasser, Schaefer
- **`hansen_data`** - Datasets from Hansen et al. 2022 for autoradiography (Receptor names and data) 
- **`macaque`** - Macaque atlas and structural connectivity, plus files from Mejias & Wang 2022 on area names, original hierarchy values, SLN etc. Also contains the Julich macaque architectonic maps and the macaque receptor density map used to extract NMDA densities per parcel.
- **`structural_connectivity_*_EBRAINS`** - EBRAINS structural connectivity matrices in Schaefer 100 and 200. Both folders also hold the tract length matrices; the Schaefer 200 ones are used to derive the conduction delays.

#### `data/processed/` - Cleaned or derived datasets

- **`macaque/`** - Processed macaque datasets such as hierarchy and SLN based on T1w:T2w images
- **(rest of the files)** - Data used for the human model (autoradiography data on AMPA, NMDA, GABA-A, hierarchy, NMDA PET receptor density, structural connectivity etc.). It also contains the files added for the revision: the principal functional connectivity gradient (`*_fc_grad.csv`), the hierarchy and SLN derived from it (`*_FC_grad_hierarchy.csv`, `*_SLN_FC_grad.csv`), and the tract lengths and conduction delays (`*_lengths.csv`, `*_delays.csv`).

### data_analysis_autorad_hierarchy/
Contains code to analyse the autoradiography data for AMPA and NMDA, and related it to cortical hierarchy based on T1w:T2w images in humans, using Schaefer 100. Also produces maps of total long range excitation / inhibition across the network. Used to complie the data analysis part of the paper (Figures 1 and 2).
#### `data_analysis_autorad_hierarchy/notebooks/` - Relevant Jupyter notebooks to produce the relationships between the three properties
#### `data_analysis_autorad_hierarchy/results/` - Contains folders where the notebooks save the autoradiography brain maps of AMPA, NMDA; total long-range excitation/inhibition; T1w:T2w and hierarchy maps. The folders are empty in the repository, since the versions of these plots that end up in the paper are the ones produced by `Figures/Figure_1.ipynb` and `Figures/Figure_2.ipynb` and saved under `Figures/saved_plots`.

### Figures/

Contains notebooks and scripts to reproduce the main figures in the paper. Each notebook has to be run sequentially (starting with Figure_1) in order to reproduce all results, as later notebooks could depend on results given by earlier notebooks. Results are saved in Figures/saved_plots (including .npy files).

- **`Figure_1.ipynb`** - Autoradiography AMPA and NMDA receptor densities and the NMDA/AMPA ratio on Schaefer 100, their relationship with the T1w:T2w cortical hierarchy, and the T1w:T2w and hierarchy brain maps. Includes a spin test that validates the receptor-hierarchy correlations against spatial autocorrelation.
- **`Figure_2.ipynb`** - Long-range projection strengths to the excitatory vs. the inhibitory populations of each node, based on SLN and the structural connectome, illustrating the counter-stream inhibitory hypothesis.
- **`Figure_3_CE.ipynb`** - Main task simulation with the autoradiography-based model (Schaefer 100), activity traces and brain plots for the cue and delay periods.
- **`Figure_3_DF.ipynb`** - Main task simulation with the PET NMDA model (Schaefer 200) at the optimal parameters, activity traces and brain plots for the cue and delay periods.
- **`Figure_4.ipynb`** - Phase diagram of G-alpha, plus the traces and brain plots for the over-active, under-active, mid-active and optimal conditions. Assumes the sweep has already been run on a cluster.
- **`Figure_5.ipynb`** - Long-range input to E vs. I with the persistent activity nodes overlaid, and delay firing rate against hierarchy and NMDA receptor density.
- **`Figure_6B.ipynb`** - Phase diagrams of feedback selectivity vs. global coupling (G), assuming the sweep was already performed on a cluster.
- **`Figure_6D_data_gen.py`** - Runs the simulations for the selectivity-G pairs selected by `Figure_6B.ipynb`, producing the firing rate and BOLD data used for decoding.
- **`Figure_6D_decoding_plots.ipynb`** - Linear decoding over time from BOLD and from firing rates, for the different feedback selectivity - G values.
- **`Figure_6E.ipynb`** - BOLD and firing rate traces for the selectivity 0.6 and selectivity 1 cases used in the decoding analysis.
- **`Supp_Figure_2.ipynb`** - Same optimal PET NMDA simulation as Figure 3, but showing the activity traces of a different set of areas and a labelled version of the average firing rate ranking plot.
- **`Supp_Figure_3.ipynb`** - Main task simulation with the model that infers AMPA from the NMDA/AMPA ratio vs. hierarchy relationship.
- **`Supp_Figure_5.ipynb`** - Distractor resistance simulation with the PET NMDA model at the optimal parameters.

#### `Figures/revision/` - Notebooks written for the revision of the manuscript, with their output saved under `Figures/revision/plots`.

- **`rebuttal_figure_FC_gradient.ipynb`** - Plots the principal functional connectivity gradient (Margulies et al., 2016) on Schaefer 100, left hemisphere.
- **`rebuttal_figure_FC_correlation_t1t1hier.ipynb`** - Correlation between the FC gradient-based hierarchy and the T1w:T2w-based hierarchy.
- **`rebuttal_figure_FC_gradient_model.ipynb`**, **`rebuttal_figure_fc_grad.ipynb`** - Main task simulation and activity panel for the model built on the FC gradient hierarchy (`models/model_pet_nmda_FC_grad`).
- **`rebuttal_figure_delays.ipynb`** - Working memory activity profile for the model with long-range conduction delays (`models/model_pet_nmda_delays`).
- **`rebuttal_figure_wc.ipynb`** - Activity panel for the Wilson-Cowan version of the network (`models/model_pet_nmda_WC`).
- **`rebuttal_figure_vary_Z.ipynb`** - Relaxes the Z homeostatic condition to check the robustness of the results (`models/model_pet_nmda_vary_Z`).
- **`rebuttal_figure_inhibition.ipynb`** - Brain plots for the run in which the persistently active areas are inhibited back to the spontaneous state, using `models/model_pet_nmda`.

### models/

Contains the different models we used to run the experiments presented in the paper and in its revision. Each model folder is self-contained, it has its own config file, source code and everything required to plot and produce results. The four models used for the main text are `model_autoradiography_nmda_ampa`, `model_pet_nmda`, `model_pet_nmda_inferred-ampa` and `model_pet_nmda_mixed_selectivity`; the remaining ones were added for the revision and are variants of `model_pet_nmda`.

#### `models/model_autoradiography_nmda_ampa/` - Model built using autoradiography data for NMDA and AMPA using Schaefer 100.
- **`config`** - Contains file that defines parameters of the model, function to update them etc.
- **`notebooks`** - Contains Jupyter notebook to run the model interactively.
- **`results`** - Folder which has figures from the main task simulated with this model, firing rates based on the interactive notebook and a plot of the connectome using Schaefer 100.
- **`scripts`** - Contains scripts to plot brain maps with the cue and delay activity, plot the connectome on Schaefer 100 and run a simulation of the main WM task.
- **`src`** - Data loading, processing and model source code.

#### `models/model_pet_nmda/` - Model built using PET NMDA data.
- **`config`** - Contains file that defines parameters of the model, function to update them etc.
- **`notebooks`** - Contains Jupyter notebooks to analyze the G-alpha parameter exploration and build the phase diagram, to plot activation maps and specific regions, and to compare the persistent activity with the data. The **`rev`** subfolder holds the phase diagram notebooks for the flattening variants explored during the revision, together with a notebook that stacks the SLN variants into a single panel.
- **`results`** - Folder which has figures from the following experiments: distractor resistance, main task with optimal paramters, main task with over and under active profiles, phase diagram of G-alpha, and a plot of the connectome. Added for the revision: **`main_task_inh`** (persistent areas driven back to the spontaneous state by extra inhibition), **`main_task_global_coupling_zero`** (main task with the global coupling set to 0), the **`g_alpha_*`** folders holding the sweeps for the flattening and SLN variants produced by `scripts_flattening`, and **`combined_panels`** with the stacked SLN variant panel.
- **`scripts`** - Contains scripts to: perform the G-alpha sweep on a cluster computer (.py and .sh files), run the distractor resistance, the optimal parameters simulation for the main task and the over/under active conditions for the phase diagram. It also contains `run_simulation_optimal_parameters_inhibit.py`, which inhibits the persistently active areas after the delay period, and `run_simulation_optimal_parameters_global_zero.py`, which runs the task with the global coupling set to zero to check whether isolated nodes can sustain persistent activity.
- **`scripts_flattening`** - Contains scripts to compute parameter sweeps for flattening of hierarchical interactions, recurrent excitation etc. 
- **`src`** - Data loading, processing and model source code.

#### `models/model_pet_nmda_inferred-ampa/` - Model built using PET NMDA data and using AMPA receptor values based on the relationship between NMDA/AMPA ratio and cortical hierarchy (obtained from autoradiography).
- **`config`** - Contains file that defines parameters of the model, function to update them etc.
- **`results`** - Folder which has figures the main task simulation.
- **`scripts`** - Contains scripts to run the main simulation with optimal parameters.
- **`src`** - Data loading, processing and model source code.

#### `models/model_pet_nmda_mixed_selectivity/` - Model built using PET NMDA data which investigates variations in feedback selectivity.
- **`config`** - Contains file that defines parameters of the model, function to update them etc.
- **`notebooks`**:
    - **`phase_diagram_selectivity_G_and_save_pairs_for_decoder.ipynb`**: Jupyter notebook to plot the phase diagram based on feedback selectivity vs. global coupling (G).
    - **`run_WM_model`**: Interactive Jupyter notebook to run a simulation where one can vary the feedback selectivity, global coupling and observe results.
- **`results`** - Contains .npy files with firing rate and BOLD information for populations A and B, generated by the file at scripts/run_simulations_for_decoder_mixed_selectivity.py. Also contains the folder with figures for the paper on decoding over time.
- **`scripts`** - Contains scripts to run the simulations with relevant feedback selectivity vs. G pairs to produce the last figure in the paper and the scripts to run and evaluate the phase space of feedback selectivity vs. G (.py and .sh files to run on a cluster).
- **`src`** - Data loading, processing and model source code.

The decoding notebooks that used to live in this folder now sit in `Figures/` as `Figure_6D_decoding_plots.ipynb` and `Figure_6E.ipynb`, so that the whole figure can be reproduced from one place.

#### `models/model_pet_nmda_FC_grad/` - Revision variant of the PET NMDA model in which the long-range connectivity is built from the principal functional connectivity gradient (Margulies et al., 2016) instead of the T1w:T2w hierarchy. It loads `schaefer200_LH_FC_grad_hierarchy.csv` and `schaefer200_LH_SLN_FC_grad.csv` and runs the main task at G = 1.25.
- **`config`**, **`src`** - Same structure as `model_pet_nmda`, with the data loading pointed at the FC gradient files.
- **`notebooks`** - Notebook to plot the activation map.
- **`results`** - Main task figures and firing rates for this variant.
- **`scripts`** - Script to run the main simulation with the optimal parameters.

#### `models/model_pet_nmda_delays/` - Revision variant of the PET NMDA model that adds long-range conduction delays. The synaptic gating variables of the excitatory populations are stored over time so that each long-range input can be read at the delayed time index, using the delays in `schaefer200_LH_delays.csv`.
- **`config`**, **`src`**, **`notebooks`**, **`scripts`** - Same structure as above, with the delayed integration implemented in `src/models.py`. As for the Wilson-Cowan variant, `notebooks/rev` is an unused copy of the phase diagram notebooks of `model_pet_nmda`.
- **`results`** - Main task figures and firing rates for this variant.

#### `models/model_pet_nmda_WC/` - Revision variant in which the network nodes are Wilson-Cowan excitatory-inhibitory pairs rather than the mean-field populations used in the rest of the paper.
- **`config`**, **`src`** - Source code with the Wilson-Cowan equations in `src/models.py`.
- **`notebooks`** - Notebooks to plot the activation map, to build the supplementary figure and to inspect the nullclines and the stability of a single node. The **`rev`** subfolder is an unused copy of the phase diagram notebooks of `model_pet_nmda`, carried over when this folder was branched off.
- **`results`** - Main task figures for this variant.
- **`scripts`** - Script to run the main simulation.

#### `models/model_pet_nmda_vary_Z/` - Revision variant used to relax the Z homeostatic condition. The model takes an extra `z_factor` that rescales the normalisation of the long-range inhibitory weights, so that the robustness of the results to violations of that condition can be tested.
- **`config`**, **`src`**, **`notebooks`**, **`scripts`**, **`scripts_flattening`** - Same structure and same set of experiments as `model_pet_nmda`, with `src/models.py` extended by the `z_factor` argument.
- **`results`** - Carries over the result folders of `model_pet_nmda` (main task, distractor resistance, over/under active, phase diagram of G-alpha, connectome and the `g_alpha_*` sweeps). The figures that actually vary Z are produced by `Figures/revision/rebuttal_figure_vary_Z.ipynb` and saved under `Figures/revision/plots/vary_Z`.

#### `models/macaque/` - Implementation of the macaque cortex model of Mejias & Wang 2022.
- **`config`** - Parameters of the macaque model.
- **`notebooks`** - Interactive Jupyter notebook to run the model.
- **`results`** - Activity traces for all areas.
- **`src_utils`** - Network and utility source code.

### notebooks/

Contains interactive Jupyter notebooks to analyse the macaque data used to validate our method, and to map Glasser to Schaefer for investigating the working-memory related areas.

- **`macaque_t1_t2.ipynb`** - Extracts T1w:T2w data for the macaque to build the cortical hierarchy and compute the SLN based on this method in order to compare it with the original results in Mejias & Wang 2022.
- **`macaque_receptor_densities.ipynb`** - Extracts NMDA receptor densities per macaque parcel from the receptor density map, using the Julich macaque architectonic maps, and saves them to `notebooks/results/macaque`.
- **`map_glasser_to_schaefer100.ipynb`**, **`map_glasser_to_schaefer200.ipynb`** - Map each region in the Glasser atlas (180 regions per hemispehre) to Schaefer 100 and Schaefer 200. This makes it easier to map the Schaefer regions to anatomical locations.
- **`schaefer_glasser_plot.ipynb`** - Builds a table with Schaefer and Glasser region names for ease of understanding. Also plots the relevant persistent activity areas (during the main task, under optimal parameters and using the PET NMDA model only) on the brain, allowing us to compare the parcels with the parcels in the Glasser parcellation (to double check that our results are correct).

### scripts/

Contains data processing scripts and aggregation of the parameter space exploration results.

- **`data_processing`** - Contains scripts to build T1w:T2w, hierarchy maps on Schaefer 100, 200; generate the structural_connectivity * (feedforward/feedback proportions) matrices for quantifying long-range excitation / inhibition to use in the data analysis part of the model; computing the SLN matrices for Schaefer 100, 200; extract NMDA receptor density based on PET for Schaefer 200; extract region names for Schaefer 100, 200 and extract structural connectivity based on diffusion-weighted MRI for Schaefer 100, 200. Added for the revision: scripts to parcellate the principal functional connectivity gradient on Schaefer 100 and 200, to compute the SLN matrix from that gradient, and to extract the tract lengths and derive the conduction delays for Schaefer 200.
- **`compile_phase_diagram_results`** - Contains script to iterate through result folders of the parameter space explorations (e.g. G-alpha, G-selectivity) and put them together in a .csv file which we can analyze more easily.
