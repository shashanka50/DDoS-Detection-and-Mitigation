# creating a class which gives all the insights of the dataset


class Insights:
    def __init__(self, df):
        self.df = df
        self.columns = df.columns
        self.shape = df.shape
        self.describe = df.describe()
        self.dtypes = df.dtypes
        self.isnull = df.isnull().sum()
        self.duplicated = df.duplicated().sum()
        self.nunique = df.nunique()
        self.columns = df.columns
        self.memory_usage = df.memory_usage()
        self.ndim = df.ndim
        self.dtypes = df.dtypes
    
    # function which prints the insights of the dataset in a formatted way
    def print_insights(self):
        print(f"Shape of the dataset: {self.shape}", end="\n\n")
        print(f"Info of the dataset: \n {self.df.info()}", end="\n\n")
        print(f"Description of the dataset: {self.describe}", end="\n\n")
        print(f"Data types of the dataset: {self.dtypes}", end="\n\n")
        print(f"Null values in the dataset: {self.isnull}", end="\n\n")
        print(f"Duplicated values in the dataset: {self.duplicated}", end="\n\n")
        print(f"Unique values in the dataset: {self.nunique}", end="\n\n")
        print(f"Columns in the dataset: {self.columns}", end="\n\n")
        print(f"Memory usage of the dataset: {self.memory_usage}", end="\n\n")
        print(f"Number of dimensions in the dataset: {self.ndim}", end="\n\n")
        print(f"Data types of the dataset: {self.dtypes}", end="\n\n")

if __name__ == "__main__":
    df = pd.read_csv("final_dataset.csv")
    insights = Insights(df)
    insights.print_insights()