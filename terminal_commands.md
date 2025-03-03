# Terminal Commands for Local on Divan PC

> Creating the datastore: python -m mllam_data_prep C:\Users\23603526\Documents\GitHub\neural-lam-Divan\era5_2020\test.datastore.yaml

> Creating the graph: python -m neural_lam.create_graph --config_path C:\Users\23603526\Documents\GitHub\neural-lam-Divan\era5_2020\test_config.yaml --name single --levels 1

> Plot the graph: python -m neural_lam.plot_graph --datastore_config_path C:\Users\23603526\Documents\GitHub\neural-lam-Divan\era5\config.yaml --graph multiscale

> Training a model: python -m neural_lam.train_model --config_path C:\Users\23603526\Documents\GitHub\neural-lam-Divan\era5\config.yaml --model graph_lam --epochs 2 --batch_size 8 --graph multiscale --ar_steps_train 1 --loss wmse --ar_steps_eval 7

> Validating a model on 2020 data: python -m neural_lam.validate_model --config_path era5_2020\test_config.yaml --load C:\Users\23603526\Documents\GitHub\neural-lam-Divan\saved_models\train-graph_lam-4x64-02_07_10-3371\min_val_loss.ckpt

# Terminal Commands for Lightning AI

> Creating the datastore: python -m mllam_data_prep /teamspace/studios/this_studio/dk-neural-lam/era5_large/era_large.datastore.yaml

> Creating the graph: python -m neural_lam.create_graph --config_path /teamspace/studios/this_studio/dk-neural-lam/era5_large/config_large.yaml --name multiscale

> Plot the graph: python -m neural_lam.plot_graph --datastore_config_path /teamspace/studios/this_studio/dk-neural-lam/era5_large/config_large.yaml --graph multiscale_D1

> Training a model: python -m neural_lam.train_model --config_path /teamspace/studios/this_studio/dk-neural-lam/era5/config.yaml --model graph_lam --epochs 2 --batch_size 8 --graph multiscale --ar_steps_train 1 --loss wmse --ar_steps_eval 7 --num_workers 16 --num_future_forcing_steps 0

python -m neural_lam.train_model --load /teamspace/studios/this_studio/dk-neural-lam/saved_models/train-graph_lam-4x64-02_26_08-0199/min_val_loss.ckpt 

> Validating a model on 2020 data: python -m neural_lam.validate_model --config_path /teamspace/studios/this_studio/dk-neural-lam/era5/config.yaml --load /teamspace/studios/this_studio/dk-neural-lam/saved_models/train-graph_lam-4x64-02_20_13-6017/min_val_loss.ckpt --ar_steps_train 3

> Validating a model using **pl.Trainer**: 

python -m neural_lam.train_model --num_workers 4 --load /teamspace/studios/this_studio/dk-neural-lam/saved_models/train-graph_lam-4x64-03_03_09-9005/min_val_loss.ckpt --eval test --batch_size 1 --ar_steps_eval 12