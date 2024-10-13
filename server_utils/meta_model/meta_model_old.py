

class MetaModel:
    def __init__(self,meta_features, total_prob = 0.9):
        self.meta_features = meta_features
        self.total_prob = total_prob

    def recommendModels(self):
        """
            implement the code to recommend models from meta model
        """

        return ['ELASTICNETCV','LinearSVR','XGBRegressor'], [0.07,0.33,0.60]
