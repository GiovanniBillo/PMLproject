# A Hidden Markov Model for exploring prediction trajectories in Transformers
Authors:
- Giovanni Billo
- Giacomo Amerio

## Description
The aim of this project was to explore probability trajectories produced by transformers in a simple sentiment analysis task to answer different questions:
- can we model the transformer's ground truth and infer when it changes state?
- which words make the transformer change state? Are they comparable to "human" switch words?
- Can the HMM "distill" the transformer's decision in a more lightweight model?

## Setup
In order to setup the environment for the correct execution of the notebooks, it is recommended to create a separate python virtual environment and install the necessary libraries there:
```bash
conda create --name pmlenv --file=env-cpu.yml
```
The files to do this (`env-cpu.yml` and 'requirements.txt') are provided in the repo.
All the project's constants are contained in `src/config.py` for easy adaptation.  

## Run 
This repository already provides the user with a trained model for loading and visualizing the results.

**Note that obtaining the data and training the model anew can take up to several hours depending on the hardware.**  
If this is your intent, uncomment the corresponding lines in `HMM.ipynb`:
```python
# hmm_surrogate_model = None
# if train_trajectories:
#     hmm_surrogate_model = HMMSurrogate() # Uses N_HMM_STATES from config
#     hmm_surrogate_model.train(train_trajectories)
    
#     # Save the trained HMM model
#     if not os.path.exists('models'):
#         os.makedirs('models')
#     hmm_surrogate_model.save_model(HMM_MODEL_PATH)
# else:
#     print("Skipping HMM training as no trajectories were loaded.")

# hmm_surrogate_model = HMMSurrogate.load_model(HMM_MODEL_PATH)
```

Once all necessary libraries are installed, open the jupyter notebooks within your favourite IDE and execute them:
example:
```bash
jupyter notebook notebooks/HMM.ipynb
```
## Structure
Jupyter notebooks are utilized to display the results. More specifically, they are divided as follows:
- `HMM.ipynb`: main notebook, trains the model if it's not already present and displays the main results of the analysis (Transition and emission matrix, visualization)
- `clustering.ipynb`: plots the embeddings of the transition tokens
- `convergence_diagnostics.ipynb`: performs convergence diagnostics checks on the model (usual Log likelihood convergence and [Posterior Predictive Check](https://cran.r-project.org/web/packages/LaplacesDemon/vignettes/BayesianInference.pdf). 
- `notebook_giacomo.ipynb`: contains further analysis on the synthesis capabilities of HMM states with respect to Transformers, using simple Multi-Layered Perceptrons.

### Credits
This project was done as part of the final exam for the probabilistic Machine Learning course by prof. Bortolussi, 2025.
