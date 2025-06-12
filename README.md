# A Hidden Markov Model for exploring prediction trajectories in Transformers
Authors:
- Giovanni Billo
- Giacomo Amerio

## Description
In this project we used and HMM to abstract the probability trajectories produced by transformers in a simple sentiment analysis task. 
The objective to answer different questions:
- can we model the transformer's ground truth and infer when it changes state?
- which words make the transformer change state? Are they comparable to "human" switch words?
- Can the HMM "distill" the transformer's decision in a more lightweight model?

## Setup
In order to setup the environment for the correct execution of the notebooks, it is recommended to create a separate python virtual environment and install the necessary libraries there:
```bash
conda create --name pmlenv --file=env.yml
```
The files to do this (`env-cpu.yml` and 'requirements.txt') are provided in the repo.
All the project's constants are contained in `src/config.py` for easy adaptation.  

## Run 
This repository already provides the user with a trained model for loading and visualizing the results. 
Moreover, the logged probability trajectories for the train split of the IMDB dataset are also provided. You can find it under `notebooks/data`.

**Note that obtaining the data and training the model anew can take up to several hours depending on the hardware.**  
If this is your intent, run the entirety of `HMM_Classification.ipynb`. It is recommended to run this code on a GPU to achieve a reasonable training time.

## Structure
Jupyter notebooks are utilized to display the results. More specifically, they are divided as follows:
- `HMM_Classification.ipynb`: main notebook, trains the model if it's not already present and displays the main results of the analysis (Transition and emission matrix, visualization)
- `clustering.ipynb`: plots the embeddings of the transition tokens
- `convergence_diagnostics.ipynb`: performs convergence diagnostics checks on the model (usual Log likelihood convergence and [Posterior Predictive Check](https://cran.r-project.org/web/packages/LaplacesDemon/vignettes/BayesianInference.pdf). 

### Credits
This project was done as part of the final exam for the probabilistic Machine Learning course by prof. Bortolussi, 2025.
