import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd #use when data is implicitly inferred
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input layer (4 features of the flower) --> Hidden Layer1 (number of neurons) -->
    # Hidden Layer1 (number of neurons) --> H2
    # (n) --> output (3 classes of iris flower)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() #instantiate our nn.module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
# Pick a manual seed for randomization
torch.manual_seed (41)
# Create an instance of model 
model= Model ()


url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

species_mapping = {
    'setosa':0,
    'versicolor':1, 
    'virginica': 2
}

#change in data type
my_df['species'] = my_df['species'].replace(species_mapping)
my_df['species'] = my_df['species'].astype(int) 

# Train Test Split! Set X, y 
X = my_df.drop ('species', axis=1)
y = my_df['species']

#convert to numpy arrays ---float
X= X.values.astype(float)
y= y.values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Converted X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor (X_test)

# Converted y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test) 

# Established criterion of model to measure error, how far off the predictions 
criterion = nn.CrossEntropyLoss()

#lr = learning rate (if error does not go down)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#train model 
#epoch 
epochs= 100 
losses = []
for i in range(epochs):
# go forward/get prediction
    y_pred = model(X_train)

    #error/loss measurement
    loss = criterion (y_pred, y_train) #pred values vs. y train

#loss tracker
    losses.append(loss.detach().numpy())
# print every 10 epoch
    if i % 10 == 0:
        print(f'Epoch {i} and loss {loss.item()}')

# back propagation: take the error rate of forward propagation and feed it back to fine tune
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print ("Training complete") 

