# Description: This file contains the metrics used to evaluate the performance of the models.
class Metrics:
    @staticmethod
    def nmss(targets, reconstructions, train_data):
        targets = targets.real
        reconstructions = reconstructions.real
        _min, _max = targets.min(), targets.max()
        n_targets = (targets - _min) / (_max - _min)
        n_train_data = (train_data - _min) / (_max - _min)
        n_reconstructions = (reconstructions - _min) / (_max - _min)
        mse = ((n_targets - n_reconstructions)**2).mean()
        norm_term = n_train_data.var(0).mean()
        nmse = mse / norm_term
        nmss = 1.0 - nmse
        return nmss
