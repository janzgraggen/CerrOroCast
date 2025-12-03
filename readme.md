# [Semester Project, Fall 2025, Environmental Computational Science and Earth Observation Laboratory (ECEO)]: 

> Orography-informed surface temperature forecasting across Europe using CERRA reanalysis data. 

> Jan Zgraggen 
> Supervised by Chang Xu, Devis Tuia

## Setup: 

1. pip or conda install Install Climatelearn according to it's propper specifications. it will be stored at : into  miniconda/...../climate_learn or venv/..../climatelearn

2. Clone [github.com/janzgraggen/climate_learn_oro](https://github.com/janzgraggen/climate_learn_oro) into  the same directory where climate learn is.

3. rename climate_learn     -> climate_learn_old
          climate_learn_oro -> climate_learn

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
python CerrOroCast/experiments/forecasting/cerra534_minimal.py  vitginr   --logname=vitginr  --vis=epoch_XXX
```

