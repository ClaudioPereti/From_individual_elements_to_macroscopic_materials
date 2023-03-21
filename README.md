# AISC
***Artificial Intelligence for SuperConductors***

This repository is a data-science project related to superconduttivity phenomena.
The project tackles both classification and regression problems and, making use of the Deep Set architecture, provide a new characterization of chemical elements to help discovery and understood superconductors.

## Set up
To assist reproducibility and ease of use we provide a dockerfile to build a container with the necessary depedencies.

```
 sudo docker image build -t aisc:latest .
 sudo docker container run -it --rm -p 8888:8888  --volume $(pwd):/AISC aisc

```
The last command will open a terminal within which we are able to reproduce the experiments conducted.

# Project Structure
The folder structure is:
```
AISC
└── project_aisc
    ├── config
    ├── data
    │   ├── external <- hosono.csv
    │   ├── processed
    │   └── raw <- supercon.csv & garbagein.csv
    ├── models
    ├── notebooks
    ├── reports
    └── src
        ├── data
        ├── features
        ├── laboratory
        ├── model
        └── utils
  ```
  In config there are  config files (yalm) to customize models and training process.<br>
  Notebooks contain notebooks to analyze the data we had and we produced during the experiments.<br>
  The code used to produce the results is stored into laboratory;

  # Run Experiments
  We can reproduce 3 experiments:
  ```
  #We can run the experiments from any folder
  #First move in the project_aisc folder
  cd project_aisc

  #Train and test the model
  python src/laboratory/train_model.py

  #Produce latent features and inspect them
  python src/laboratory/latent_space.py

  #Compare difference features strategy
  python src/laboratory/compare_features_strategy.py

  ```
  With the appropriate flags is possible to use customized model and save the results.

  # Analyze Experiments
We can both do eda on SuperCon data or on experiment results
```
#Move inside notebook
cd notebook/

#Open the browser and pick up the desired exploration
jupyter-nbclassic

#Change kernel to python369 from kernel tab
```
