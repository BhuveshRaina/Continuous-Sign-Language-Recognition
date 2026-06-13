import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        # Attention layer to score each frame
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (Batch, Seq_Len, Hidden_Size)
        attn_scores = self.attention(lstm_output) # (Batch, Seq_Len, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # Normalize weights over sequence length
        
        # Multiply LSTM outputs by attention weights and sum over sequence
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector

class CSLModel(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=256, num_classes=108, num_layers=2):
        super(CSLModel, self).__init__()
        
        # Upgraded Feature Extractor with LayerNorm and GELU
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4) # Increased dropout for better generalization
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.4
        )
        
        # New Attention Mechanism
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Upgraded Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        lstm_out, _ = self.lstm(x)
        
        # Use Attention instead of mean pooling!
        x = self.attention(lstm_out) 
        
        out = self.classifier(x)
        return out