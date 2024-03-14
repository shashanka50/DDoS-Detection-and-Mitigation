# Naive-bayes classifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from trainer.trainer import Trainer
import matplotlib.pyplot as plt


class NB_Trainer(Trainer):
    def __init__(self):
        # initialize the parent class
        super().__init__()
        """
        Load the dataset
        """
        print("Loading dataset ...")
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')

    def flow_training(self):
        """
        Train the Naive Bayes model
        """
        print("Flow Training ...")
        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -1].values
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        # Feature scaling
        # sc = StandardScaler()
        # X_flow_train = sc.fit_transform(X_flow_train)
        # X_flow_test = sc.transform(X_flow_test)

        # Train the model
        self.model = GaussianNB()
        self.model.fit(X_flow_train, y_flow_train)

        # Predict the test set results
        y_flow_pred = self.model.predict(X_flow_test)

        # Make the confusion matrix
        self.conf_matrix = confusion_matrix(y_flow_test, y_flow_pred)
        self.accuracy = accuracy_score(y_flow_test, y_flow_pred)

        # Visualize the confusion matrix
        plt.matshow(self.conf_matrix, cmap='Blues')
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        print(self.conf_matrix)
        print(self.accuracy)

        # Save the model
        self.save_model()
        print("Model saved successfully")

    def save_model(self):
        """
        Save the model using pickle
        """
        import pickle
        with open('C:\\Users\\Shashanka G\\Desktop\\7th sem\\Project Work\\New folder\\saved_models\\NB_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
        print("Model saved successfully")


if __name__ == "__main__":
    nb = NB_Trainer()
    nb.flow_training()