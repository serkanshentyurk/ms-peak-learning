import numpy as np
import matplotlib.pyplot as plt
from .helpers import find_nearest_idx, make_image
import plotly.express as px

from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap


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
    
    
def generate_custom_cmap(k):
    tab20_cmap = get_cmap('tab20', k)
    colors = tab20_cmap(np.arange(k))
    custom_cmap = ListedColormap(colors)

    return colors, custom_cmap


    
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