### Results reproduction

Each command has to be run from folder `inverse_rl_dexterous_hand/training`.
The results can be found for each run in the `Runs` folder. Each job name is printed on the beginning of the run in the console.

In each command provide as `<env>`:
- 'relocate' for the Object relocation task
- 'pen' for the In-hand manipulation task
- 'hammer' for the Task usage task

`<env>` defaults to relocate. 

#### Table 1:
The results will be available in `Runs` folder with `DAPG_based_on` in name.

For AIRL:  
`python training.py env <env> IRL.visible_indices Null --configs IRL_into_DAPG`

For our method w/o noise samples:  
`python training.py env <env> --configs IRL_into_DAPG`

For our method with noise samples:  
`python training.py env <env> IRL.noise_samples True  --configs IRL_into_DAPG`   

#### Table 2:
Provide beta parameter as `IRL.normalization_lr`.  

Example:  
`python training.py env <env> IRL.noise_samples True IRL.normalization_lr 0.001 --configs IRL`


#### Fig. 5:
Our method:  
`python training.py env <env> IRL.noise_samples True --configs IRL`  

AIRL:  
`python training.py env <env> IRL.visible_indices Null --configs IRL`

Manual reward:  
`python training.py env <env> --configs DAPG`

### Results plotting

The results can be plotted using the provided `plot_results.py` script in folder `inverse_rl_dexterous_hand/utils`  
To this end, provide the run names to plot in the variable `result_paths` in `plot_results.py` script.
