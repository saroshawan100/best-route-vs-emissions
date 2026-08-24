'''
The model we are building is based on the STAEformer (Spatio-Temporal Adaptive Embedding Transformer). It is highly effective for 
traffic forecasting because it replaces heavy Graph Neural Networks with "Adaptive Embeddings". This means it learns the physical 
map of the Bay Area roads purely by observing the data over time.

Instead of forcing the model to calculate massive N times N matrices to figure out how every street connects, 
this class gives every single one of the 2,352 sensors a set of trainable parameters. As the model trains, 
it automatically groups sensors that share traffic patterns.
'''
import torch
import torch.nn as nn

class STAdaptiveEmbedding(nn.Module):
    """
    Learns spatial and temporal dependencies natively without needing a pre-defined graph.
    """
    def __init__(self, num_nodes, embed_dim, seq_len):
        super().__init__()
        # Spatial adaptive embedding: Learns a unique fingerprint for all 2,352 sensors
        self.node_emb = nn.Parameter(torch.randn(1, 1, num_nodes, embed_dim))
        
        # Temporal adaptive embedding: Learns patterns for the 12 time steps
        self.time_emb = nn.Parameter(torch.randn(1, seq_len, 1, embed_dim))

    def forward(self, x):
        # x shape: (Batch, Seq_len, Nodes, Dim)
        return x + self.node_emb + self.time_emb

class STAEformer(nn.Module):
    def __init__(self, num_nodes=2352, in_dim=1, seq_len=12, horizon=12, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        
        #Feature Projection
        self.input_proj = nn.Linear(in_dim, embed_dim)
        
        # Spatio-Temporal Adaptive Embeddings
        self.st_embedding = STAdaptiveEmbedding(num_nodes, embed_dim, seq_len)
        
        #Vanilla Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        #Output Projection (Predicting the future horizon)
        self.output_proj = nn.Linear(embed_dim, horizon)

    def forward(self, x):
        """
        x: (Batch, Seq_len, Nodes, Features)
        """
        B, T, N, F = x.shape
        
        # Map raw speed data to the higher-dimensional embedding space
        x = self.input_proj(x)
        
        # Inject the physical map and time awareness
        x = self.st_embedding(x)
        
        # GPU Optimization: To prevent Out-Of-Memory errors with 2,352 nodes, 
        # we treat (Batch * Nodes) as the batch size, focusing attention purely on the temporal sequence.
        # The spatial relationships are heavily carried by the node embeddings.
        x = x.transpose(1, 2).reshape(B * N, T, -1) 
        
        # Pass through the Transformer layers
        x = self.transformer_encoder(x)
        
        # Reshape back to original dimensions
        x = x.reshape(B, N, T, -1)
        
        # Extract the representation of the very last time step to predict the future
        last_step_repr = x[:, :, -1, :]  
        
        # Project to the output horizon (12 future steps)
        out = self.output_proj(last_step_repr) 
        
        # Reshape output to match labels: (Batch, Horizon, Nodes)
        out = out.transpose(1, 2)  
        
        return out