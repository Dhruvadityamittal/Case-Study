import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class QuestionClassifier(nn.Module):
    def __init__(self, n_classes, BATCH_SIZE, MAX_LEN, pretrained_model):
        super(QuestionClassifier, self).__init__()
        self.MAX_LEN = MAX_LEN
        # self.BATCH_SIZE = BATCH_SIZE
        
        # Load pre-trained BERT model
        self.pretrained_model = pretrained_model
        
                
        # Define an LSTM layer
        self.LSTM = nn.LSTM(self.pretrained_model.config.hidden_size, MAX_LEN, 1, batch_first=True)
        
        # # Define additional convolutional layers
        self.conv1 = torch.nn.Conv1d(in_channels=MAX_LEN, out_channels=32, kernel_size=3, padding='same')
        # self.pooling = torch.nn.AdaptiveAvgPool1d()
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        
        # Define a fully connected layer
        self.Linear = nn.Linear(16*MAX_LEN, n_classes)

    def forward(self, input_ids, attention_mask):
        
        # Initialize hidden and cell states for LSTM
        self.h0 = torch.randn(1, input_ids.shape[0], self.MAX_LEN).to(device)
        self.c0 = torch.randn(1, input_ids.shape[0], self.MAX_LEN).to(device)


        # Disable gradient calculation for BERT to save memory and computation
        with torch.no_grad():
            output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state from BERT output
        hidden_states = output.last_hidden_state
        
        
        # Apply the convolutional layer
        # output = self.cnn(hidden_states)
        
        # Pass the output through the LSTM layer
        output, (hn, cn) = self.LSTM(hidden_states, (self.h0, self.c0))
        
        
        # Apply the additional convolutional layers
        output = self.conv1(output)
        # output = self.pooling(self.pooling)
        output = self.conv2(output)
        # output = self.pooling(self.pooling)
        
        # Flatten the output for the fully connected layer
        flatten = output.view(output.shape[0], -1)
        
        # Pass the flattened output through the fully connected layer
        dense1 = self.Linear(flatten)
        
        return dense1
