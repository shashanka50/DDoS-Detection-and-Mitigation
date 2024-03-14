# K-nearest neighbors algorithm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from trainer.trainer import Trainer
import matplotlib.pyplot as plt


class KNN_Trainer(Trainer):
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
        Train the KNN model with hyper-parameter tuning
        """
        print("Flow Training ...")
        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -1].values
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        # Feature scaling
        sc = StandardScaler()
        X_flow_train = sc.fit_transform(X_flow_train)
        X_flow_test = sc.transform(X_flow_test)

        # Define the hyperparameters to tune
        param_grid = {
            # 'n_neighbors': [5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1)

        # Perform hyperparameter search
        grid_search.fit(X_flow_train, y_flow_train)

        # Get the best parameters and model
        self.best_params = grid_search.best_params_
        print("Best parameters saved ...")
        self.model = grid_search.best_estimator_

        # Evaluate the model
        y_flow_pred = self.model.predict(X_flow_test)
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)
        print("Accuracy: ", accuracy_score(y_flow_test, y_flow_pred))

        # Visualize the confusion matrix
        plt.matshow(cm, cmap=plt.cm.gray)
        plt.show()
        # save the image
        plt.savefig('confusion_matrix.png')

        print("Training complete ...")
        self.save_model()
        print("Model saved ...")

    def save_model(self):
        """
        Save the model using pickle
        """
        import pickle
        with open('KNN_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)


if __name__ == "__main__":
    knn = KNN_Trainer()
    knn.flow_training()
    knn.save_model()