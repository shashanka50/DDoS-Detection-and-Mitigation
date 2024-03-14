class Trainer:

    def __init__(self):
        """
        Load the dataset
        """
        import pandas as pd
        self.flow_dataset = pd.read_csv('C:\\Users\\Shashanka G\\Desktop\\7th sem\\Project Work\\New folder\\final_dataset_1.csv')
        self.model = None
        self.best_params = None

    def flow_training(self):
        # overriding for different classification modules
        pass

    def save_model(self):
        # overriding for different classification modules
        pass 