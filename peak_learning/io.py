import numpy as np

def load_data(path_load, dataset = 'M2'):
    data_original = np.load(path_load + '/' + dataset + "_matrix.npy")
    mz_vector = np.load(path_load + '/' + dataset +  "_mz_vector.npy")
    row2grid = np.load(path_load + '/' + dataset +  "_row2grid.npy")

    data = np.copy(data_original)
    data[data <= 0] = 0
    residual = np.copy(data_original)
    residual[residual > 0] = 0    
    return data, residual, mz_vector, row2grid
