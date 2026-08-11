import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap


def guassian_input_check(x, chunk_count):
    if isinstance(x, int) or isinstance(x, float):
        return np.repeat(x, chunk_count)
    elif np.array(x).shape[0] == 1:
        return np.repeat(x, chunk_count)
    else:
        return np.array(x)
        
def gaussian_mixture_determine_n_component(mz_array, n_component_max = 30, n_repeat_experiment= 10, max_iter = 100, n_init = 10, covariance_type = 'diag'):
    aic_results = np.empty((n_repeat_experiment, n_component_max-1))
    bic_results = np.empty((n_repeat_experiment, n_component_max-1))
    
    for repeat in range(n_repeat_experiment):
        for n_component in range(1,n_component_max):
            model = GaussianMixture(n_components = n_component, max_iter = max_iter, n_init = n_init, covariance_type = covariance_type)
            model.fit(mz_array)
            aic_results[repeat, n_component-1] = model.aic(mz_array)
            bic_results[repeat, n_component-1] = model.bic(mz_array)

    return aic_results, bic_results

def plot_aic_bic(obj, aic_result_list, bic_result_list, vertical_line, suptitle = False):
    fig, axs = plt.subplots(len(aic_result_list), 2, figsize = (15, 10))
    for chunk in range(len(aic_result_list)):
        ax = axs[chunk][0]
        aic_results = aic_result_list[chunk]
        bic_results = bic_result_list[chunk]
        
        ax.plot(range(1, aic_results.shape[1]+1),aic_results.mean(axis = 0), 'r', alpha = 1, lw = 1, label = 'AIC mean')
        for i in range(len(aic_results)):
            if i == 0:
                ax.plot(range(1, aic_results.shape[1]+1),aic_results[i], '-.r', alpha = 0.2, lw = 1, label = 'AIC')
            else:
                ax.plot(range(1, aic_results.shape[1]+1),aic_results[i], '-.r', alpha = 0.2, lw = 1)
                
        ax.plot(range(1, aic_results.shape[1]+1),bic_results.mean(axis = 0), 'b', alpha = 1, lw = 1, label = 'BIC mean')
        for i in range(len(bic_results)):
            if i == 0:
                ax.plot(range(1, aic_results.shape[1]+1),bic_results[i], '-.b', alpha = 0.2, lw = 1, label = 'BIC')
            else:
                ax.plot(range(1, aic_results.shape[1]+1),bic_results[i], '-.b', alpha = 0.2, lw = 1)

        ax.axvline(x = vertical_line[chunk], color = 'k', linestyle = '--', lw = 0.7, label = f'Vertical Line - {vertical_line[chunk]}')
        # ax.axvline(x = obj.chunks[chunk][1], color = 'r', linestyle = '--', lw = 0.7, label = f'Peak Detection Optimal Component: {obj.chunks[chunk][1]}')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('AIC - BIC Scores')
        ax.set_title(f'AIC - BIC Scores for Gaussian Mixture Model - Chunk {chunk+1}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax = axs[chunk][1]
        ax.plot(obj.chunks[chunk][0], obj.chunks[chunk][1], 'o', markersize = 5, label = 'Correlation')
        ax.plot(obj.peaks[0][chunk], np.ones(obj.peaks[0][chunk].shape[0]), 'o', markersize = 5, label = f'Detected Peaks - {obj.peaks[0][chunk].shape[0]}')
        ax.set_xlabel('m/z')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Chunk {chunk+1}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if suptitle:
        plt.suptitle('AIC - BIC Scores for Gaussian Mixture Model - All Chunks', fontsize = 16)
    plt.tight_layout()
    plt.show()

    
def generate_custom_cmap(k):
    tab20_cmap = get_cmap('tab20', k)
    colors = tab20_cmap(np.arange(k))
    custom_cmap = ListedColormap(colors)

    return colors, custom_cmap

