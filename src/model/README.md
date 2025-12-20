# Model Code
To test temperature prediction from historical data only use data_preprocess.py.

To test microcontroller data and historical data in conjunction, use process_historical_data.py. This will create a deployable model but it is not very accurate due to limited microcontroller data.

To test the deployable model using just microcontroller training data, use finalmodelformicrocontrllerdeployment.py. Then use getparams.py and norm_params.npz (a new such file will be created) to print parameters which will be put into the Arudino script. Also, create a model.h.
