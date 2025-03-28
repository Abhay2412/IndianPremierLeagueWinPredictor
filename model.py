import torch
import torch.nn
import pandas 
import numpy
from sklearn.model_selection import train_test_split
from dataLoader import loadMatchesData

MODEL_PATH = "./models/cricketModel.pt"

class CricketWinPredictor(torch.nn.Module):
    def __init__(self, numberOfTeams, embedDim=8, hiddenDim=16):
        super(CricketWinPredictor, self).__init__()
        # Need to do the embedding layer to convert the team indices into dense vectors 
        # Embedding is basically like a vector representation of the teams these can be values, objects like text images and even audio
        # Hidden layers are what make neural networks deep and enable them to learn complex data representations
        # The hidden layers help transforming inputs into something that the output layer can use 
        self.embedding = torch.nn.Embedding(numberOfTeams, embedding_dim=embedDim)
        # Then we do the connected layers after concatenating the team embeddings 
        self.fc1 = torch.nn.Linear(embedDim * 2, hiddenDim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hiddenDim, 1) # This is the single output for binary classification 