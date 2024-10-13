from kerasbeats import prep_time_series
from sklearn.model_selection import train_test_split


class PrepareData:
    def __init__(self, data, train_size=0.67):
        self.data = data
        self.train_freq = train_size

    def train_test_split(self):
        self.data.sort_index(inplace=True)
        self.data.loc[:, 'Target'] = self.data['Target'].astype("float")
        X, y = prep_time_series(self.data, lookback=7, horizon=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=self.train_freq)
        return X_train, X_test, y_train, y_test
