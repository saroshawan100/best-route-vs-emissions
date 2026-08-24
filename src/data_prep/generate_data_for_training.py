
'''
This script is part of the data preparation pipeline for time series forecasting tasks. It reads raw HDF5 (gba_his_2019.h5 and gba_his_2019.h5) data, processes it into 
sequences suitable for model training, and saves the resulting datasets in compressed NPZ format. The generated datasets include 
training, validation, and test splits, which will be used for model development and evaluation.
'''

    
import os
import gc
import argparse
import numpy as np
import pandas as pd

def generate_train_val_test(args):
    data_list = []
    
    # Loop through and load all provided files
    for file_path in args.input_files:
        print(f"Loading HDF5 data from: {file_path}")
        df = pd.read_hdf(file_path)
        data_list.append(df.values)
        
    # Combine and downcast to float16 to save massive amounts of RAM
    print("\nCombining data and downcasting to float16...")
    data = np.concatenate(data_list, axis=0).astype(np.float16)

    # Ensure shape is 3D: (num_time_steps, num_sensors, num_features)
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    num_samples, num_nodes, num_features = data.shape
    print(f"Combined data shape: {num_samples} time steps, {num_nodes} sensors, {num_features} feature(s)")

    seq_len = args.seq_len
    horizon = args.horizon

    min_t = seq_len
    max_t = num_samples - horizon
    total_samples = max_t - min_t + 1

    # Compute Train / Val / Test sizes
    num_train = int(total_samples * args.train_ratio)
    num_val = int(total_samples * args.val_ratio)
    
    # Define start and end indices for each split
    splits = [
        ('train', min_t, min_t + num_train),
        ('val', min_t + num_train, min_t + num_train + num_val),
        ('test', min_t + num_train + num_val, max_t + 1)
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    # Process and save one split at a time to prevent RAM overflow
    for cat, start_idx, end_idx in splits:
        num_split_samples = end_idx - start_idx
        print(f"\nProcessing '{cat}' split ({num_split_samples} samples)...")
        
        # Pre-allocate arrays (Much more memory efficient than lists)
        x_arr = np.empty((num_split_samples, seq_len, num_nodes, num_features), dtype=np.float16)
        y_arr = np.empty((num_split_samples, horizon, num_nodes, num_features), dtype=np.float16)
        
        for i, t in enumerate(range(start_idx, end_idx)):
            x_arr[i] = data[t - seq_len : t]
            y_arr[i] = data[t : t + horizon]
            
        output_path = os.path.join(args.output_dir, f"{cat}.npz")
        print(f"Compressing and saving {cat}.npz (this may take a minute)...")
        np.savez_compressed(output_path, x=x_arr, y=y_arr)
        print(f" - Saved: X shape = {x_arr.shape}, Y shape = {y_arr.shape}")
        
        # Force Python to free up the RAM before moving to the next split
        del x_arr
        del y_arr
        gc.collect()

    print(f"\nSuccessfully generated memory-optimized training data in '{args.output_dir}'!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', default=['data/ca/GBA/gba_his_2018.h5', 'data/ca/GBA/gba_his_2019.h5'], help='Paths to raw .h5 files')
    parser.add_argument('--output_dir', type=str, default='data/ca/GBA/processed', help='Output directory for .npz files')
    parser.add_argument('--seq_len', type=int, default=12, help='Sequence length (historical windows)')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon (future windows)')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Training set split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set split ratio')

    args = parser.parse_args()
    generate_train_val_test(args)