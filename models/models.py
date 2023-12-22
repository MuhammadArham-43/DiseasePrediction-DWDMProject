from abc import abstractmethod
from typing import Union

import numpy as np

class BaseModel:
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        pass


class DecisionTree(BaseModel):
    def __init__(
        self,
        criterion: Union["gini", "entropy"] = "entropy"
        ) -> None:
        super().__init__()
        
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(criterion=criterion)
        print(f"USING DECISION TREE CLASSISFIER WITH {criterion.upper()} LOSS")
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class SVMClassifierModel(BaseModel):
    def __init__(
        self,
        kernel: Union["rbf", "linear", "poly", "sigmoid"] = "rbf"   
    ) -> None:
        super().__init__()
        
        from sklearn.svm import SVC
        self.model = SVC(kernel=kernel)
        print(f"USING SUPPORT VECTOR CLASSIFIER WITH {kernel.upper()} FUNCTION")
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class NeuralNetworkModel(BaseModel):
    def __init__(
        self,
        num_features: int,
        num_classes: int    
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        
        import tensorflow as tf
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.Dense(units=num_classes, activation="sigmoid"),
        ])

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        print("USING DENSE NEURAL NETWORK")
    
    def fit(self, X, y):
        from tensorflow.keras.utils import to_categorical
        y_one_hot = to_categorical(y, num_classes=self.num_classes)
        self.model.fit(X, y_one_hot, epochs=20, batch_size=32)
    
    def predict(self, X):
        predictions_probs = self.model.predict(X)
        predictions = np.argmax(predictions_probs, axis=1)
        return predictions


class NaiveBayesModel(BaseModel):
    def __init__(
        self,
    ):
        super().__init__()
        
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB()
        print("USING GUASSIAN NAIVE BAYES")
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class RandomForestModel(BaseModel):
    def __init__(
        self,
        num_estimators:int=10
    ) -> None:
        super().__init__()

        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=num_estimators)
        print("USING RANDOM FOREST ENSEMBLE MODEL")
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
