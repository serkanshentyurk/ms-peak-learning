import os
import numpy as np
import matplotlib.pyplot as plt

from .helpers import make_image, find_nearest_idx
from .io import load_data

class MSIData:
    def __init__(self, path):
        '''
        obj.correlated_insulin has 4 columns: m/z value, correlation, correlated with which insulin (0 both, 1 insulin1, 2 insulin2) and expressed in nonislets (boolean)
        '''
        dataset = path[-2:]
        self.data_all, self.residual, self.mz_vector, self.row2grid = load_data(path, dataset = dataset)
        self.data_all[self.data_all < 0] = 0
        
        # cluster_indexes.npy is produced by the mapping step (IsletMap). Load it if
        # it already exists so PeakModel/validation can use the labels. IsletMap is
        # what *generates* the labels, so it must be able to construct without it.
        path_index = os.path.join(path, 'cluster_indexes.npy')
        if os.path.exists(path_index):
            with open(path_index, 'rb') as f:
                self.cluster_labels = np.load(f)
            self.cluster_labels_map = make_image(self.row2grid, self.cluster_labels)
        else:
            self.cluster_labels = None
            self.cluster_labels_map = None

        self.data = np.empty((self.data_all.shape[1], 228, 165))
        for mz_index in range(self.data_all.shape[1]):
            self.data[mz_index] = make_image(self.row2grid, self.data_all[:,mz_index])
        self.outside_map = np.sum(self.data, axis=0) < 1
        self.data[self.data < 0] = 0
        
        self.x_max = np.max(self.row2grid[:,0])
        self.x_min = np.min(self.row2grid[:,0])
        self.delta_x = self.x_max - self.x_min + 1 
        self.y_max = np.max(self.row2grid[:,1])
        self.y_min = np.min(self.row2grid[:,1])
        self.delta_y = self.y_max - self.y_min + 1
        
        self.insulin_1_mz = None
        self.insulin_2_mz = None
        self.insulin_1_index = None
        self.insulin_2_index = None
        
        self.minimum_variance = None
        self.minimum_variance_index = None

        self.correlation_matrix = None
        self.correlation_threshold_1 = None
        self.correlation_threshold_2 = None
        
        self.correlated_1_map = None
        self.correlated_1_mz = None
        self.correlated_1_corr = None

        self.correlated_2_map = None
        self.correlated_2_mz = None
        self.correlated_2_corr = None
        
        self.correlated_both_mz = None
        self.correlated_both_1_indices = None
        self.correlated_both_1_corr = None

        self.correlated_both_2_indices = None
        self.correlated_both_2_corr = None

        self.correlated_only_1_mz = None
        self.correlated_only_1_indices = None
        self.correlated_only_1_corr = None

        self.correlated_only_2_mz = None
        self.correlated_only_2_indices = None
        self.correlated_only_2_corr = None

        self.correlated_all_mz = None
        self.correlated_all_1_corr = None
        self.correlated_all_2_corr = None
        
        self.peaks_1_indices = None
        self.peaks_2_indices = None
        self.peaks_1 = None
        self.peaks_2 = None
                
        self.chunk_borders = None
        self.chunk_count = None
        self.chunks = None
        
                
        self.edges_magnitude = None
        self.circle_map_bin = None
        self.circle_map_cv = None
        
        self.data_to_cluster = None
        self.data_to_cluster_x_range = None
        self.data_to_cluster_y_range = None
        self.data_to_cluster_reshaped = None
        self.data_to_cluster_clustered_kmeans = None
        self.data_to_cluster_clustered_hdbscan = None
        
    def plot_ion_image(self, mz):
        plt.grid(False)
        plt.imshow(self.data[self.mz_index(mz)], cmap = 'hot')

    def mz_index(self, mz):
        return find_nearest_idx(self.mz_vector, mz)

    def plot_islet_mz(self, vertical_1 = 5800, vertical_2 = 6000, x_limit = [5500, 6500], set_mz = False, larger_than = 400, sample_index = 0, suptitle = False):
        vertical_1_index = self.mz_index(vertical_1)
        vertical_2_index = self.mz_index(vertical_2)
        
        fig = plt.figure(figsize=(10, 8))

        # Plot in the first row spanning both columns
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        x = np.where(self.data[self.mz_index(5800)] > larger_than)[0][sample_index]
        y = np.where(self.data[self.mz_index(5800)] > larger_than)[1][sample_index]
        ax1.plot(self.mz_vector,self.data[:,x,y])
        ax1.set_xlim(x_limit)
        ax1.axvline(vertical_1, c='r', lw=1, ls='--', label = f'Candidate $Ins1$ - {vertical_1}')
        ax1.axvline(vertical_2, c='g', lw=1, ls='--', label = f'Candidate $Ins2$ - {vertical_2}')
        ax1.set_xlabel('m/z')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Intensities of m/z values at (x = {y}, y = {x})')
        ax1.legend()

        # Plot in the second row, first column
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        plt.grid(False)
        image_1 = ax2.imshow(self.data[vertical_1_index], cmap='hot')
        ax2.set_title(f'Ion Image - m/z: {vertical_1}')

        # Adding colorbar
        colorbar_1 = plt.colorbar(image_1, ax=ax2)
        colorbar_1.set_label('Intensity')  # You can customize the label as needed

        # Plot in the second row, second column
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        plt.grid(False)
        image_2 = ax3.imshow(self.data[vertical_2_index], cmap='hot')
        ax3.set_title(f'Ion Image - m/z: {vertical_2}')

        # Adding colorbar
        colorbar_2 = plt.colorbar(image_2, ax=ax3)
        colorbar_2.set_label('Intensity')  # You can customize the label as needed

        # Add title
        if suptitle:
            plt.suptitle('Candidate $Ins1$ and $Ins2$ and Corresponding Ion Images', fontsize=14)

        # Adjust layout manually
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, hspace=0.3, wspace=0.7)
        # Show the plot
        plt.show()
        
        if set_mz:
            self.set_insulin(insulin_1_mz = vertical_1, insulin_2_mz = vertical_2)
        
    def set_insulin(self, insulin_1_mz = 5800, insulin_2_mz = 6278):
        self.insulin_1_mz = insulin_1_mz
        self.insulin_2_mz = insulin_2_mz
        self.insulin_1_index = self.mz_index(insulin_1_mz)
        self.insulin_2_index = self.mz_index(insulin_2_mz)
        