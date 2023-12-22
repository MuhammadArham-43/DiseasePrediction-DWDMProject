import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

LABEL_COLUMN_NAME = "prognosis"

class DiseaseDataset:
    def __init__(
        self, 
        csv_path: "str",
        do_attribute_reduction: bool=True,
        pca_components: int = 40
    ) -> None:
        
        assert pca_components > 0 and pca_components <= 132, "PCA COMPONENTS CAN BE BETWEEN 1 to 132"
        
        self._dataset_path = csv_path
        self.data = pd.read_csv(self._dataset_path)
        self.label_encoder = LabelEncoder()

        self.y = np.array(self.data[LABEL_COLUMN_NAME])
        self.X = np.array(self.data.drop([LABEL_COLUMN_NAME], axis=1))
        
        self.preprocess(do_attribute_reduction=do_attribute_reduction, num_components=pca_components)

        
    def preprocess(self, do_attribute_reduction=True, num_components:int=40):
        self.y = self.label_encoder.fit_transform(self.y)        
        if do_attribute_reduction:
            self.X = PCA(n_components=num_components).fit_transform(self.X)
    
    def get_data(self):
        return self.X, self.y

    def get_num_classes(self):
        return len(self.label_encoder.classes_)

    def get_unqiue_classes(self):
        return self.label_encoder.classes_

    def get_num_attributes(self):
        return self.X.shape[1]