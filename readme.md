# [Semester Project, Fall 2025, Environmental Computational Science and Earth Observation Laboratory (ECEO)]: 

> Orography-informed surface temperature forecasting across Europe using CERRA reanalysis data. 

> Jan Zgraggen 
> Supervised by Chang Xu, Devis Tuia

## Setup: 

1. pip or conda install Install [Climate_learn](https://github.com/aditya-grover/climate-learn) according to it's propper specifications found it its installation instructions. It will be stored at : into  miniconda/...../climate_learn or venv/..../climatelearn if you use conda or venv as your package manager. 

2. Clone [github.com/janzgraggen/climate_learn_oro](https://github.com/janzgraggen/climate_learn_oro) into  the same directory where climate learn is.

3. delete/rename climate_learn     -> climate_learn_old (or delete)
   rename        climate_learn_oro -> climate_learn

> Maybe cloning is sufficient...

## Run Comands
The following comands are used for obtaining the respective experiments. 

### Training The Explicit model 
in the __main__ comment in the desired model that you want to train then run:
```bash
python CerrOroCast/SIDM_models/convolution_based/SIDM_convolution_mapper.py 
```

### Forecasting: 
```bash
python CerrOroCast/experiments/cerra534_minimal.py              vitginr   --logname=vitginr
```

### Vizsulaisatuin; 
```bash
python CerrOroCast/experiments/cerra534_minimal.py  vitginr   --logname=vitginr  --vis=epoch_XXX
```

