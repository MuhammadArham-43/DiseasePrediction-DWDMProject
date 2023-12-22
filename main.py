from dataset import DiseaseDataset
from models import DecisionTree, SVMClassifierModel, NeuralNetworkModel, RandomForestModel, NaiveBayesModel
from utils import calulate_metrics, display_confusion_matrix

from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    
    DATA_CSV_PATH = "data/diseasedata.csv"
    DATA_CSV_PATH = os.path.join(os.getcwd(), DATA_CSV_PATH)    
    
    data = DiseaseDataset(DATA_CSV_PATH, do_attribute_reduction=True)
    X, y = data.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)    
    
    # -- Uncomment Any Model -- #
    # model = DecisionTree(criterion="gini")
    # model = SVMClassifierModel(kernel="rbf")
    # model = NeuralNetworkModel(num_features=data.get_num_attributes(), num_classes=data.get_num_classes())
    # model = RandomForestModel(num_estimators=20)
    model = NaiveBayesModel()
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    calulate_metrics(y_test, predictions)
    display_confusion_matrix(y_test, predictions, classes=data.get_unqiue_classes())