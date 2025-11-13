# [Semester Project, Fall 2025, Environmental Computational Science and Earth Observation Laboratory (ECEO)]: 

> Orography-informed surface temperature forecasting across Europe using CERRA reanalysis data. 

> Jan Zgraggen 
> Supervised by Chang Xu, Devis Tuia

## Run Comands

#### --> NEW Minimal
for running (with hyperparam args)

 `python CerrOroCast/experiments/cerra534_minimal.py --bs=16 --logname=geofar_v2 dataset/CERRA-534/ geofar_v2  6`

for running minimaly

 `python CerrOroCast/experiments/cerra534_minimal.py   geofar_v2   --logname=geofar_v2`

#### --> New Visualizing
for running visualization
 `python CerrOroCast/experiments/forecasting/cerra534_minimal.py  geofar_v2   --logname=geofar_v2 --vis=epoch_XXX`
