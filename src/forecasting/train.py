import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from dataset import get_dataloaders
from staeformer import STAEformer

def main():
    # Setup Hardware Acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    #  Load the Formatted Data
    data_dir = "C:\\Users\\saros\\Downloads\\best-route-vs-emissions\\data\\ca\\GBA\\processed"
    print("Loading datasets (this will load train.npz, val.npz, and test.npz)...")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=64)

    # \Initialize the STAEformer Model
    # We have 500 sensors, predicting 1 feature (speed), using 12 past steps to predict 12 future steps
    print("Initializing STAEformer model...")
    model = STAEformer(num_nodes=500, in_dim=1, seq_len=12, horizon=12, embed_dim=64).to(device)
    
    # Define Loss Function and Optimizer
    # Mean Absolute Error (L1Loss) is the standard metric for traffic forecasting
    criterion = nn.L1Loss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Execute Training Loop
    epochs = 10
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move data tensors to the GPU (or CPU)
            x, y = x.to(device), y.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass: Predict future speeds
            outputs = model(x)
            
            # Calculate error
            loss = criterion(outputs, y)
            
            # Backward pass: Update model weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print status every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Train Loss: {avg_train_loss:.4f} ---")

   
    import os
    os.makedirs("models", exist_ok=True)
    save_path = "models/staeformer_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights successfully saved to: {save_path}")
    # --------------------------------------
            

if __name__ == "__main__":
    main()