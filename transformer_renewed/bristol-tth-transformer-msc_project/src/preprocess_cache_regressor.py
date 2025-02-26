import os
import torch
import pickle
from datetime import datetime
from preprocessing import (
    load_from_parquet,
    remove_negative_events,
    create_multiclass_labels,
    apply_reweighting_per_class_multiclass,
    awkward_to_inputs_parallel_multiclass,
    get_event_level,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def get_preprocessed_dataset_regressor(cache_path="cached_dataset_regressor.pkl", test_size=0.2, random_state=42, n_processes=8, target_length=10):
    if os.path.exists(cache_path):
        print("Loading preprocessed (regressor) data from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
        
    print("Processing parquet files...")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"./model_training/AOTransformer_{current_time}"
    os.makedirs(outdir, exist_ok=False)

    # Specify dataset files
    path = "/cephfs/dice/projects/CMS/Hinv/ml_datasets_ul_241111/ml_inputs_UL{year}/{dataset}.parquet"
    datasets = ['TTToSemiLeptonic']
    years = ['2018']
    files = [path.format(year=year, dataset=dataset) for dataset in datasets for year in years]

    # Preprocessing
    df = load_from_parquet(files,regions=[0,6])
    df = remove_negative_events(df)
    
    #df["target"] = create_multiclass_labels(df["dataset"])
    target_regressor = "GenMET_pt"
    df["target_regressor"] = df[target_regressor]




    apply_reweighting_per_class_multiclass(df)
    reweighting = torch.Tensor(df['weight_training'].values)
    weight_nominal_tensor = torch.Tensor(df['weight_nominal'].values)
    
    
    X, y, pad_mask = awkward_to_inputs_parallel_multiclass(df, n_processes=n_processes, target_length=target_length)
    event_level = get_event_level(df)



    input_MET_vals = torch.Tensor(df['MET_pt'].values)
    input_MET_phi_vals = torch.Tensor(df['MET_phi'].values)
    


    (train_X, val_X, train_y, val_y, train_weights, val_weights,
     train_mask, val_mask, train_event, val_event, train_weight_nominal, val_weight_nominal, train_MET_phi, val_MET_phi, train_target, val_target) = train_test_split(
        X, 
        y, 
        reweighting, 
        pad_mask, 
        event_level, 
        weight_nominal_tensor,
        input_MET_phi_vals,
        target_regressor,
        test_size=0.2, 
        random_state=42,
    )


###
#### NEED TO SCALE THE target_regressor ###
###


    train_dataset = TensorDataset(train_X, train_y, train_weights, train_mask, train_event, train_MET_phi, train_target)
    valid_dataset = TensorDataset(val_X, val_y, val_weights, val_mask, val_event, val_MET_phi, val_target)

    data = {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "val_weight_nominal": val_weight_nominal,
        'X': X,
        'y': y,
        'pad_mask': pad_mask,
        'event_level': event_level,
        'reweighting': reweighting,
        'weight_nominal_tensor': weight_nominal_tensor,
        'reweighting': reweighting,
        'train_X': train_X,
        'val_X': val_X,
        'train_y': train_y,
        'val_y': val_y,
        'train_weights': train_weights,
        'val_weights': val_weights,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'train_event': train_event,
        'val_event': val_event,
        'train_weight_nominal': train_weight_nominal,
        'val_weight_nominal': val_weight_nominal,
        'outdir': outdir,
        'train_MET_phi': train_MET_phi,
        'val_MET_phi': val_MET_phi,
        'train_target': train_target,
        'val_target': val_target,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data




