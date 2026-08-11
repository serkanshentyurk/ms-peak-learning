import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.express as px
from scipy.stats import norm
from sklearn.cluster import KMeans

def load_data(path_load, dataset = 'M2'):
    data_original = np.load(path_load + '/' + dataset + "_matrix.npy")
    mz_vector = np.load(path_load + '/' + dataset +  "_mz_vector.npy")
    row2grid = np.load(path_load + '/' + dataset +  "_row2grid.npy")

    data = np.copy(data_original)
    data[data <= 0] = 0
    residual = np.copy(data_original)
    residual[residual > 0] = 0    
    return data, residual, mz_vector, row2grid


def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def make_ion_image(mz_value, data, mz_vector, row2grid, save = False, path_save = None):
    mz_vector = np.ndarray.flatten(mz_vector)
    decimals = str(mz_value)[::-1].find('.')
    if decimals==-1:
        decimals=0
    index = np.where(np.round(mz_vector, decimals)==mz_value)[0]
    if len(index) == 0:
        index = find_nearest_idx(mz_vector, mz_value)
    else:
        index = index[0]
    result_2D = make_image(row2grid, data[:,index])
    plt.grid(False)
    plt.imshow(result_2D)
    plt.colorbar()
    rounded_val3 = np.round(mz_vector[index],3)
    rounded_val0 = np.round(mz_vector[index],0)
    formatted_val0 = "{:.0f}".format(rounded_val0)
    plt.title("m/z = "+ str(rounded_val3))
    if save:
        plt.savefig(f"{path_save}mz_{formatted_val0}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def make_ion_image_px(mz_value, data, mz_vector, row2grid, save = False, path_save = None):
    mz_vector = np.ndarray.flatten(mz_vector)
    decimals = str(mz_value)[::-1].find('.')
    if decimals==-1:
        decimals=0
    index = np.where(np.round(mz_vector, decimals)==mz_value)[0]
    if len(index) == 0:
        index = find_nearest_idx(mz_vector, mz_value)
    else:
        index = index[0]
        
    result_2D = make_image(row2grid, data[:,index])
    rounded_val3 = np.round(mz_vector[index],3)
    rounded_val0 = np.round(mz_vector[index],0)
    formatted_val0 = "{:.0f}".format(rounded_val0)
    fig = px.imshow(result_2D, title = "m/z = "+ str(rounded_val3), color_continuous_scale='viridis')
    # px.colorbar()
    if save:
        plt.savefig(f"{path_save}mz_{formatted_val0}.png", dpi=300, bbox_inches='tight')
    fig.show()
        
        
        
def compute_median(v):
    non_zero_elements = v[np.nonzero(v)]
    if len(non_zero_elements) > 0:
        return np.median(non_zero_elements)
    else:
        return 0

def compute_mean(v):
    non_zero_elements = v[np.nonzero(v)]
    if len(non_zero_elements) > 0:
        return np.mean(non_zero_elements)
    else:
        return 0
    

def grid2row(x,y, row2grid):
    xmax = np.max(row2grid[:,0])
    xmin = np.min(row2grid[:,0])
    ymax = np.max(row2grid[:,1])
    ymin = np.min(row2grid[:,1])
    # print(xmax+1, ymax+1, xmin, ymin, xmax-xmin, ymax-ymin)
    grid2row = np.zeros((xmax+1, ymax+1), dtype=int) + np.nan
    for r, c in enumerate(row2grid):
        grid2row[c[0], c[1]] = r
    return int(grid2row[x+xmin,y+ymin])
    

def make_image(row2grid, spatial_i):
    xmax = np.max(row2grid[:,0])
    xmin = np.min(row2grid[:,0])
    ymax = np.max(row2grid[:,1])
    ymin = np.min(row2grid[:,1])

    image_matrix = np.zeros([xmax-xmin+1,ymax-ymin+1])
    image_matrix = image_matrix - 2
    k = 0
    for e in row2grid:
        image_matrix[e[0]-xmin,e[1]-ymin] = spatial_i[k]
        k+=1
    return image_matrix

def r_from_z(z):
    r = (np.exp(2*z)-1) / (np.exp(2*z)+1)
    return r 
def r_to_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def find_threshold_z(obj, r, significance_level=0.05):
    z1 = r_to_z(r)
    n = obj.data.shape[1] * obj.data.shape[2]
    # Calculate standard error
    standard_error = 1 / (n ** 0.5)
    
    # Find the critical z-value for the desired significance level
    critical_z_value = norm.ppf(1 - significance_level)
    
    # Calculate the minimum z2 value
    min_z2 = z1 + critical_z_value * standard_error
    min_r2 = r_from_z(min_z2)
    return min_r2

def fisher_z_test(r1, n1, r2, n2 = None, verbose = True):
    if n2 == None:
        n2 = n1
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_diff = (z1 - z2) / se_diff

    p_value = norm.cdf(z_diff)
    
    if verbose:
        print(f"Z-Difference: {z_diff}")
        print(f"P-Value: {p_value}")

        if p_value < 0.05:
            print(f"\nThe correlation r2 is significantly smaller than r1:\t p-value = {round(p_value,4)}")
        else:
            print("The difference is not statistically significant.")

    return z_diff, p_value

def find_indices_for_cluster(obj, k_1 = 3, k_2 = 3):
    kmeans_1 = KMeans(n_clusters = k_1)
    kmeans_1.fit(obj.correlated_insulin_1[:,0].reshape(-1,1))
    labels_1 = kmeans_1.labels_

    first_element_indices_dict_1 = {}

    for i, label in enumerate(labels_1):
        if label not in first_element_indices_dict_1:
            first_element_indices_dict_1[label] = i
            
    first_element_indices_1 = list(first_element_indices_dict_1.values())
    
    kmeans_2 = KMeans(n_clusters = k_2)
    kmeans_2.fit(obj.correlated_insulin_2[:,0].reshape(-1,1))
    labels_2 = kmeans_2.labels_
    
    first_element_indices_dict_2 = {}
    
    for i, label in enumerate(labels_2):
        if label not in first_element_indices_dict_2:
            first_element_indices_dict_2[label] = i
    
    first_element_indices_2 = list(first_element_indices_dict_2.values())
    
    return first_element_indices_1, first_element_indices_2

def return_candidate_peaks(obj, plot = False):
    random = np.sort((np.vstack((obj.gmm_chunks[0].means_, obj.gmm_chunks[1].means_, obj.gmm_chunks[2].means_)).reshape(-1))).reshape(-1,1)
    known = obj.all_candidate_peaks.reshape(-1,1)
    if plot:
        xlim = [[2800, 4500],[5600, 6100], [11000,12000]]
        fig, axs = plt.subplots(3, 1,figsize = (10,8))

        for i in range(3):
            ax = axs[i]
            ax.plot(random, np.ones(random.shape), 'o', label = 'Random Peaks', ms = 3)
            ax.plot(known, np.ones(known.shape) - 0.15, 'o', label = 'Known Peaks', ms = 3)
            ax.set_xlabel('$m/z$')
            ax.set_xlim(xlim[i])
            ax.set_ylim(0.7, 1.1)
            ax.set_yticks([0.85, 1])
            ax.set_yticklabels(['Means\nRandom Peaks', 'Means\nKnown Peaks'])
            ax.set_title(f'Detected Peaks - Cluster {i+1}')
        plt.tight_layout()
        plt.show()
    return random, known