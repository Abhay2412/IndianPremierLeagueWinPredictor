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
    def forward(self, expectedShape): # Basically we are defining how the input data will flow through the model 
        # expected shape of the teams we are going to do which will be 2 teams 
        firstTeam = expectedShape[:, 0].long() # Each row contains two integers representing the indices of the two teams
        secondTeam = expectedShape[:, 1].long()
        embedded1 = self.embedding(firstTeam)
        embedded2 = self.embedding(secondTeam)
        expectedShapeConcat = torch.cat([embedded1, embedded2], dim=1) # Concatenates the two embeddings along the feature dimension.
        expectedShapeHidden = self.fc1(expectedShapeConcat) # Applies the ReLU activation function to the hidden layer output.
        expectedShapeAct = self.relu(expectedShapeHidden)
        resultProbability = self.fc2(expectedShapeAct)
        return resultProbability

def preprocessData(dfMatches: pandas.DataFrame):
    """Helps with preprocessing the data 
    Going to drop the matches with no result and maps the team names to numeric codes
    Will also make 1 if the first team wins otherwise 0 like a binary

    Parameters
    ----------
    dfMatches : pandas.DataFrame
        All of the matches 
    """
    dfMatches = dfMatches.dropna(subset=['winner'])