import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data():

    df = pd.read_csv("../data/titanic/train.csv")  # Ensure the path is correct

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    y_processed = y.values

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Client(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config=None):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        # Update model parameters
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        X_tensor = torch.FloatTensor(self.X_train)
        y_tensor = torch.FloatTensor(self.y_train).view(-1, 1)

        for epoch in range(5):  # Number of epochs
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = nn.BCELoss()(output, y_tensor)
            loss.backward()
            self.optimizer.step()

        accuracy = ((output.round() == y_tensor).float().mean()).item()  # Calculate accuracy
        return self.get_parameters(), len(self.X_train), {"loss": loss.item(), "accuracy": accuracy}  # Return metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_test)
            y_tensor = torch.FloatTensor(self.y_test).view(-1, 1)
            output = self.model(X_tensor)
            loss = nn.BCELoss()(output, y_tensor)
            accuracy = ((output.round() == y_tensor).float().mean()).item()
        return float(loss), len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    # Preprocess the data
    X_train, y_train, X_test, y_test = preprocess_data()

    model = TitanicModel()
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=Client(model, X_train, y_train, X_test, y_test))
