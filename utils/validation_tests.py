import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import ttest_ind, mannwhitneyu
from utils.basic_utils import find_nearest_idx
from utils.method_utils import generate_custom_cmap
import umap

def find_significant_peaks(obj, peaks, alpha = 0.01, parametric = True, verbose = True):
    significant_mz = []
    non_significant_mz = []
        
    for mz_interest in peaks:
        mz_index_interest = find_nearest_idx(obj.mz_vector, mz_interest)
        array1 = obj.data[mz_index_interest,obj.cluster_map_altered == 0]
        array2 = obj.data[mz_index_interest,obj.cluster_map_altered == 1]

        if parametric:
            # Perform Welch's t-test
            t_statistic, p_value = ttest_ind(array2, array1, equal_var=False, alternative = 'greater')
        else:
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(array2, array1, alternative = 'greater')
        # Check the p-value
        if p_value < alpha: # nonparametric
            significant_mz.append(mz_interest[0])
            # print(f"m/z = {round(mz_interest[0],2)} is significantly different.\nP-value: {p_value:.4f}\n")
        else:
            non_significant_mz.append(mz_interest[0])
            if verbose:
                print(f"*** m/z = {round(mz_interest[0],2)} is NOT significantly different.\nP-value: {p_value:.4f} ***\n") 
    if verbose:
        print(f'There are {len(non_significant_mz)} nonsignificant m/z values.')
        print(f'There are {len(significant_mz)} significant m/z values.')
        
    return significant_mz, non_significant_mz

def plot_candidate_peak_dist(obj, candidates, candidate_index = 0):

    mz_interest = candidates[candidate_index]
    
    mz_index_interest = find_nearest_idx(obj.mz_vector, mz_interest)
    array1 = obj.data[mz_index_interest,obj.cluster_map_altered == 0]
    array2 = obj.data[mz_index_interest,obj.cluster_map_altered == 1]

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.grid(False)
    plt.imshow(obj.cluster_map_altered, cmap='viridis', alpha = 1)
    plt.imshow(obj.data[mz_index_interest], cmap='coolwarm', alpha = 0.7)
    plt.title('Intensity Distribution of $m/z$ = {}'.format(round(mz_interest,2)))
    plt.subplot(1,2,2)
    plt.grid(True)
    h1 = plt.hist(array1, bins = 100, log = True, alpha = 0.7, label = 'Non-Islet')
    h2 = plt.hist(array2, bins = 100, log = True, alpha = 0.7, label = 'Islet')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Intensity Distribution of $m/z$ = {}'.format(round(mz_interest,2)))
    plt.show()
    
def plot_all_candidates(random, known, sliding):
    xlim = [[2800, 4500],[5600, 6100], [11000,12000]]
    fig, axs = plt.subplots(3, 1,figsize = (10,8))

    for i in range(3):
        ax = axs[i]
        ax.plot(random, np.ones(random.shape), 'o', label = 'Random Peaks', ms = 3)
        ax.plot(known, np.ones(known.shape) - 0.15, 'o', label = 'Known Peaks', ms = 3)
        ax.plot(sliding, np.ones(sliding.shape) - 0.3, 'o', label = 'Chris Peaks', ms = 3)
        ax.set_xlabel('$m/z$')
        ax.set_xlim(xlim[i])
        ax.set_ylim(0.55, 1.1)
        ax.set_yticks([0.7,0.85, 1])
        ax.set_yticklabels(['Means\nSliding Window', 'Means\nKnown Peaks', 'Means\nRandom Peaks'])
        ax.set_title(f'Detected Peaks - Cluster {i+1}')
    plt.tight_layout()
    plt.show()
    
def compare_arrays(array_1, array_2, delta, verbose = True, same_method = False):
    common_values_1 = []
    common_values_2 = []
    array_1_only = []
    array_2_only = []
    
    for value1 in array_1:
        found = False
        for value2 in array_2:
            if abs(value1 - value2) <= delta:
                common_values_1.append(value1)
                common_values_2.append(value2)
                found = True
                break
        if not found:
            array_1_only.append(value1)
    
    for value2 in array_2:
        found = False
        for value1 in array_1:
            if abs(value1 - value2) <= delta:
                found = True
                break
        if not found:
            array_2_only.append(value2)
            
    common_values_1 = np.sort(np.unique(common_values_1)).reshape(-1)
    common_values_2 = np.sort(np.unique(common_values_2)).reshape(-1)
    array_1_only = np.sort(np.unique(array_1_only)).reshape(-1)
    array_2_only = np.sort(np.unique(array_2_only)).reshape(-1)
        
    if same_method:
        candidates = np.sort(np.unique(np.concatenate((common_values_1, array_1_only))))
    else:
        candidates = common_values_1
    
    all_candidates = np.sort(np.unique(np.concatenate((common_values_1, common_values_2, array_1_only, array_2_only))))
    
    if verbose:
        print("\nNum values from array_1:", len(common_values_1))
        print("Num values from array_2:", len(common_values_2))
        print("Num only in array 1:", len(array_1_only))
        print("Num only in array 2:", len(array_2_only))
    return (
        all_candidates, candidates,
        np.unique(common_values_1),
        np.unique(common_values_2),
        np.unique(array_1_only),
        np.unique(array_2_only)
    )
    
def compare_output_candidate(random_gmm, known_means_gmm, sliding_window_candidates, delta = 10, verbose = True):
    all_candidates, candidates_corr, common_1_corr, common_2_corr, only_array_1_corr, only_array_2_corr = compare_arrays(random_gmm, 
                                                                                                     known_means_gmm, 
                                                                                                     delta, 
                                                                                                     verbose = verbose, 
                                                                                                     same_method = True)
    all_candidates, candidates_final, common_1_final, common_2_final, only_array_1_final, only_array_2_final = compare_arrays(candidates_corr, 
                                                                                                          sliding_window_candidates, 
                                                                                                          delta, 
                                                                                                          verbose = verbose, 
                                                                                                          same_method = False)
    
    return all_candidates, candidates_final

def plot_dist_given_mz(obj, mz_interest):
    mz_index_interest = obj.mz_index(mz_interest)
    array1 = obj.data[mz_index_interest, obj.cluster_map_altered == 0]
    array2 = obj.data[mz_index_interest, obj.cluster_map_altered == 1]

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.grid(False)
    plt.imshow(obj.cluster_map_altered, cmap='viridis', alpha = 1)
    plt.imshow(obj.data[mz_index_interest], cmap='coolwarm', alpha = 0.7)
    plt.title('Intensity Distribution of $m/z$ = {}'.format(round(mz_interest,2)))
    plt.subplot(1,2,2)
    plt.grid(True)
    h1 = plt.hist(array1, bins = 100, log = False, alpha = 0.7, label = 'Non-Islet')
    h2 = plt.hist(array2, bins = 100, log = False, alpha = 0.7, label = 'Islet')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Intensity Distribution of $m/z$ = {}'.format(round(mz_interest,2)))
    
def gaussian(x, mean, std_dev, amplitude):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

def calculate_correlated_distributions(obj, candidate_mean, plot = False):
    model_means = np.concatenate((obj.filter_models[0][0].means_.reshape(-1), obj.filter_models[1][0].means_.reshape(-1), obj.filter_models[2][0].means_.reshape(-1)))
    model_var = np.concatenate((obj.filter_models[0][0].covariances_.reshape(-1), obj.filter_models[1][0].covariances_.reshape(-1), obj.filter_models[2][0].covariances_.reshape(-1)))
    positions = np.searchsorted(model_means, candidate_mean)
    candidate_var = model_var[positions]
    
    x = obj.mz_vector
    y = np.zeros(x.shape)
    for comp in range(candidate_mean.shape[0]):
        amplitude = obj.correlation_matrix[obj.insulin_1_index][obj.mz_index(candidate_mean[comp])]
        if amplitude < 0:
            amplitude = 0
        mean = candidate_mean[comp]
        variance = candidate_var[comp]
        std_dev = np.sqrt(variance)
        y += gaussian(x, mean, std_dev, amplitude)
    y[y >1] = 1
    if plot:
        plt.figure(figsize = (10,4))
        plt.plot(x, y, label = 'Correlated Distribution')
        plt.xlabel('$m/z$')
        plt.ylabel('Correlation')
        plt.xlim(2500,7500)
    return y

def remove_artefacts(obj, candidates, plot = False):
    to_be_removed = calculate_correlated_distributions(obj, candidates, plot = plot)
    removed_data = obj.data.copy()
    to_be_removed_corr = np.matmul(obj.data[obj.insulin_1_index, obj.cluster_map_altered == 1].reshape(-1,1),to_be_removed.reshape(1,-1)).T
    to_be_removed_corr[to_be_removed_corr < 0] = 0
    removed_data[:, obj.cluster_map_altered == 1] = obj.data[:, obj.cluster_map_altered == 1] - to_be_removed_corr
    removed_data[removed_data < 0] = 0
    islet_original = obj.data[:, obj.cluster_map_altered == 1]
    islet_removed = removed_data[:,obj.cluster_map_altered == 1]
    if plot:
        insulin_artefacts_mean = np.mean(islet_original, axis = 1) -  np.mean(islet_removed, axis = 1)
        plt.figure(figsize = (10,4))
        plt.plot(obj.mz_vector, insulin_artefacts_mean, label = 'Artefacts', alpha = 0.8)
        plt.xlabel('$m/z$')
        plt.ylabel('Intensity')
        plt.xlim(2500,7500)
        plt.show()
    return islet_original, islet_removed

def plot_umap_all(obj, embedding, labels, area_x = [1, 1.8], area_y = [1.1, 2.2]):
    colors, cmap = generate_custom_cmap(4)
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c = labels, cmap=cmap, s=5)
    plt.xlim(-8,5)
    plt.axvline(x = area_x[0])
    plt.axvline(x = area_x[1])
    plt.axhline(y = area_y[0])
    plt.axhline(y = area_y[1])
    cbar = plt.colorbar(scatter, ticks=[-0.5, 0.5, 1.5, 2.5])
    cbar.ax.set_yticklabels(['-1', '0', '1', '2'])
    # plt.title('UMAP projection of the All data', fontsize=15)
    plt.show()
    weird_map = np.logical_and(np.logical_and(np.logical_and(embedding[:,0] > area_x[0], embedding[:,0] < area_x[1]), embedding[:,1] > area_y[0]), embedding[:,1] < area_y[1]).reshape(obj.cluster_map_altered.shape)
    return weird_map

def umap_all_data(obj, plot = True, area_x = (1, 1.8), area_y = (1.1, 2.2)):
    temp_data = obj.data.copy()
    temp_data = temp_data.reshape(14000,-1).T
    labels = obj.cluster_map.reshape(-1)

    np.random.seed(0)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(temp_data)
    weird_map = None
    if plot:
        weird_map = plot_umap_all(obj, embedding, labels, area_x = area_x, area_y = area_y)
    
    return embedding, labels, weird_map

def compare_unusual_umap_part(obj, unusual_map):
    plt.figure(figsize = (10,4))
    plt.plot(obj.mz_vector,np.mean(obj.data[:,obj.cluster_map_altered == 0], axis = 1), label = 'Non islet', alpha = 0.8)
    plt.plot(obj.mz_vector,np.mean(obj.data[:,obj.cluster_map_altered == 1], axis = 1), label = 'Islet', alpha = 0.8)
    plt.plot(obj.mz_vector,np.mean(obj.data[:,unusual_map], axis = 1), label = 'Area of Interest', alpha = 0.8)
    plt.legend()
    plt.xlabel('$m/z$', fontsize=14)
    plt.ylabel('Mean Intensity', fontsize=14)
    # plt.title('Mean Intensity of Different Maps', fontsize=16)
    plt.xlim(5500,7000)
    plt.show()
    
def plot_umap_islet(embedding_1, embedding_2, labels, pixel = True):
    if pixel:
        plt.figure(figsize=(4, 4))
        plt.scatter(embedding_1[:,0], embedding_1[:,1], c = labels, cmap = 'Spectral', s = 5, label = 'Original')
        plt.show()
        plt.figure(figsize=(4, 4))
        plt.scatter(embedding_2[:,0], embedding_2[:,1], c = labels, cmap = 'Spectral', s = 5, label = 'Removed')
        plt.show()
    else:
        plt.figure(figsize=(10, 4))
        scatter1 = plt.scatter(embedding_1[:, 0], embedding_1[:, 1], c=labels, cmap='Spectral', s=5, label='Original')
        plt.colorbar()
        plt.title('UMAP - Original $m/z$ Data of Islets')

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()

        plt.figure(figsize=(10, 4))
        scatter2 = plt.scatter(embedding_2[:, 0], embedding_2[:, 1], c=labels, cmap='Spectral', s=5, label='Removed')
        plt.colorbar()
        plt.title('UMAP - Artefacts Removed $m/z$ Data of Islets')

        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)
        plt.show()


def umap_islet(obj, islet_original, islet_removed, plot = False, pixel = True):
    if pixel:
        labels = obj.cluster_map[obj.cluster_map_altered == 1].reshape(-1)
        np.random.seed(0)
        reducer = umap.UMAP()
        embedding_original = reducer.fit_transform(islet_original.T)
        embedding_removed = reducer.transform(islet_removed.T)
    else:
        labels = obj.mz_vector
        np.random.seed(0)
        reducer = umap.UMAP()
        embedding_original = reducer.fit_transform(islet_original)
        embedding_removed = reducer.transform(islet_removed)
    if plot:
        plot_umap_islet(embedding_original, embedding_removed, labels, pixel = pixel)
    return embedding_original, embedding_removed, labels

def create_colours(data_3d):
    # define 3D color space
    norm_R = mcolors.Normalize(vmin=0, vmax=255)
    norm_G = mcolors.Normalize(vmin=0, vmax=255)
    norm_B = mcolors.Normalize(vmin=0, vmax=255)

    norm_R.autoscale(data_3d[:, 0])
    norm_G.autoscale(data_3d[:, 1])
    norm_B.autoscale(data_3d[:, 2])

    norm_embedding = list(zip(norm_R(data_3d)[:, 0], norm_G(data_3d)[:, 1], norm_B(data_3d)[:, 2]))
    
    return norm_embedding

def umap_non_islet(obj, plot = True):
    data_all = obj.data.copy()
    data_all = data_all.reshape(14000,-1).T

    np.random.seed(0)
    redu = umap.UMAP(n_components = 3)
    embedding_all = redu.fit_transform(data_all)
    res2d_all = create_colours(embedding_all)
    umap_all = np.array(res2d_all).reshape(228, 165, 3)
    umap_all[obj.cluster_map_altered == -1] = 0


    data_nonislet = obj.data.copy()
    data_nonislet[:,obj.cluster_map_altered == 1] = 0
    data_nonislet = data_nonislet.reshape(14000,-1).T

    np.random.seed(0)
    embedding_non_islet = redu.fit_transform(data_nonislet)
    res2d_non_islet = create_colours(embedding_non_islet)
    umap_non = np.array(res2d_non_islet).reshape(228, 165, 3)
    umap_non[obj.cluster_map_altered == 1] = 0
    umap_non[obj.cluster_map_altered == -1] = 0

    if plot:
        plt.grid(False)
        plt.imshow(umap_all)
        plt.show()
        plt.grid(False)
        plt.imshow(umap_non)
        plt.show()
    return umap_all, umap_non