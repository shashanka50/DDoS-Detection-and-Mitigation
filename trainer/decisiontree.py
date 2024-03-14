# Decision Tree Trainer

# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from trainer.trainer import Trainer

import matplotlib.pyplot as plt
import pandas as pd


class DT_Trainer(Trainer):
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
        Train the decision tree model with hyper-parameter tuning
        """
        print("Flow Training ...")
        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -1].values
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        # Define the hyperparameters to tune
        param_grid = {
            'max_depth': [3, 5, None],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'criterion': ['gini', 'entropy']
        }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1)

        # Perform hyperparameter search
        grid_search.fit(X_flow_train, y_flow_train)

        # Get the best parameters and model
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        # Evaluate the model
        y_flow_pred = self.model.predict(X_flow_test)
        print("------------------------------------------------------------------------------")
        print("Best parameters found:", self.best_params)
        print("Confusion matrix:")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)
        acc = accuracy_score(y_flow_test, y_flow_pred)
        print("Success accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        print("Fail accuracy = {0:.2f} %".format(fail*100))
        print("------------------------------------------------------------------------------")

        # Visualize the confusion matrix
        x = ['TP', 'FP', 'FN', 'TN']
        plt.title("Decision Tree")
        plt.xlabel('Predicted class')
        plt.ylabel('Number of flows')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x, y, color="#e0d692", label='DT')
        plt.legend()
        plt.show()
        # Save the image 
        plt.savefig('C:\\Users\\Shashanka G\\Desktop\\7th sem\\Project Work\\New folder\\DT.png')

    def save_model(self):
        """
        Save the model to a file
        """
        print("Saving model ...")
        import pickle
        with open('C:\\Users\\Shashanka G\\Desktop\\7th sem\\Project Work\\New folder\\saved_models\\dt_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
        print("Model saved to dt_model.pkl")


if __name__ == "__main__":
    # create the object of the class
    dt = DT_Trainer()
    # call the flow_training method
    dt.flow_training()
    # call the save_model method
    dt.save_model()