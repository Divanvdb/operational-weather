# Terminal Commands for Local on Divan PC

> Creating the datastore: python -m mllam_data_prep C:\github\operational-weather\era_test\era_test.datastore.yaml

> Creating the graph: python -m neural_lam.create_graph --config_path C:\github\operational-weather\era_test\config_test.yaml 

> Plot the graph: python -m neural_lam.plot_graph --datastore_config_path C:\Users\23603526\Documents\GitHub\neural-lam-Divan\era5\config.yaml --graph multiscale

> Training a model: python -m neural_lam.train_model --config_path C:\Users\23603526\Documents\GitHub\neural-lam-Divan\era5\config.yaml --model graph_lam --epochs 2 --batch_size 8 --graph multiscale --ar_steps_train 1 --loss wmse --ar_steps_eval 7

> Validating a model on testing data: python -m neural_lam.train_model --eval test --load C:\github\operational-weather\model_weights\min_val_loss.ckpt

In order:

python -m mllam_data_prep C:\github\operational-weather\era_test\era_test.datastore.yaml
python -m neural_lam.create_graph --config_path C:\github\operational-weather\era_test\config_test.yaml 
python -m neural_lam.train_model --eval test --load C:\github\operational-weather\model_weights\min_val_loss.ckpt
