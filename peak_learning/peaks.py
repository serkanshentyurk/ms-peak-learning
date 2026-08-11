import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks as sp_find_peaks, savgol_filter
from scipy.interpolate import interp1d

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .core import MSIData
from .stats import find_threshold_z
from .viz import plot_aic_bic


def gaussian_input_check(x, chunk_count):
    if isinstance(x, int) or isinstance(x, float):
        return np.repeat(x, chunk_count)
    elif np.array(x).shape[0] == 1:
        return np.repeat(x, chunk_count)
    else:
        return np.array(x)


def gaussian_mixture_determine_n_component(mz_array, n_component_max=30, n_repeat_experiment=10,
                                           max_iter=100, n_init=10, covariance_type='diag'):
    aic_results = np.empty((n_repeat_experiment, n_component_max - 1))
    bic_results = np.empty((n_repeat_experiment, n_component_max - 1))
    for repeat in range(n_repeat_experiment):
        for n_component in range(1, n_component_max):
            model = GaussianMixture(n_components=n_component, max_iter=max_iter,
                                    n_init=n_init, covariance_type=covariance_type)
            model.fit(mz_array)
            aic_results[repeat, n_component - 1] = model.aic(mz_array)
            bic_results[repeat, n_component - 1] = model.bic(mz_array)
    return aic_results, bic_results


class PeakModel(MSIData):
    def __init__(self, path):
        super().__init__(path)

    def find_minimum_variance_correlation(self, verbose = True):
        
        if self.insulin_1_index == None or self.insulin_2_index == None:
            print('Please run find_insulin_peaks first!')
            return
        # variance
        min_var = 10000000000
        var_ind = -1
        for i in range(self.data.shape[0]):
            varr = np.var(self.data[i])
            if varr < min_var:
                min_var = varr
                var_ind = i
        if verbose:
            print(f'Minimum variance: {round(min_var,4)}, index: {var_ind}')
        self.minimum_variance = min_var
        self.minimum_variance_index = var_ind
        
        # correlation
        correlation_ref_1 = np.corrcoef(self.data[self.insulin_1_index].reshape(-1), 
                                        self.data[self.minimum_variance_index].reshape(-1))[0,1] 
        correlation_ref_2 = np.corrcoef(self.data[self.insulin_2_index].reshape(-1), 
                                        self.data[self.minimum_variance_index].reshape(-1))[0,1] 
        
        self.minimum_correlation_with_insulin_1 = find_threshold_z(self, correlation_ref_1)
        self.minimum_correlation_with_insulin_2 = find_threshold_z(self, correlation_ref_2)
        
        if verbose:
            print(f'Min Variance m/z correlation with $Ins2$: {round(correlation_ref_1,4)}')
            print(f'Min Varaince m/z correlation with $Ins1$: {round(correlation_ref_2,4)}')
            
        
    def pearson_correlation(self, plot = True):
        self.correlation_matrix = np.corrcoef(self.data.reshape(self.data.shape[0],-1), rowvar = True)
        
        
        correlation_threshold_1 = self.minimum_correlation_with_insulin_1
        if correlation_threshold_1 < 0.1:
            self.correlation_threshold_1 = 0.1
        else:
            self.correlation_threshold_1 = correlation_threshold_1
    
        correlation_threshold_2 =  self.minimum_correlation_with_insulin_2
        if correlation_threshold_2 < 0.1:
            self.correlation_threshold_2 = 0.1
        else:
            self.correlation_threshold_2 = correlation_threshold_2
        
        self.correlated_1_map = self.correlation_matrix[self.insulin_1_index] >= self.correlation_threshold_1
        self.correlated_1_mz = self.mz_vector[self.correlated_1_map]
        self.correlated_1_corr = self.correlation_matrix[self.insulin_1_index][self.correlated_1_map]

        self.correlated_2_map = self.correlation_matrix[self.insulin_2_index] >= self.correlation_threshold_2
        self.correlated_2_mz = self.mz_vector[self.correlated_2_map]
        self.correlated_2_corr = self.correlation_matrix[self.insulin_2_index][self.correlated_2_map]

        correlated_both_1_combined = np.intersect1d(self.correlated_1_mz, self.correlated_2_mz, return_indices = True)
        self.correlated_both_mz = correlated_both_1_combined[0]
        self.correlated_both_1_indices = correlated_both_1_combined[1]
        self.correlated_both_1_corr = self.correlated_1_corr[self.correlated_both_1_indices]

        correlated_both_2_combined = np.intersect1d(self.correlated_2_mz, self.correlated_1_mz, return_indices = True)
        self.correlated_both_2_indices = correlated_both_2_combined[1]
        self.correlated_both_2_corr = self.correlated_2_corr[self.correlated_both_2_indices]

        self.correlated_only_1_mz = np.setdiff1d(self.correlated_1_mz, self.correlated_2_mz)
        self.correlated_only_1_indices = np.where(np.isin(self.correlated_1_mz, self.correlated_only_1_mz))[0]
        self.correlated_only_1_corr = self.correlated_1_corr[self.correlated_only_1_indices]

        self.correlated_only_2_mz = np.setdiff1d(self.correlated_2_mz, self.correlated_1_mz)
        self.correlated_only_2_indices = np.where(np.isin(self.correlated_2_mz, self.correlated_only_2_mz))[0]
        self.correlated_only_2_corr = self.correlated_2_corr[self.correlated_only_2_indices]

        self.correlated_all_mz = np.concatenate((np.concatenate((self.correlated_only_1_mz, self.correlated_only_2_mz)), self.correlated_both_mz))
        self.correlated_all_1_corr = np.concatenate((np.concatenate((self.correlated_only_1_corr, self.correlated_only_2_corr)), self.correlated_both_1_corr))
        self.correlated_all_2_corr = np.concatenate((np.concatenate((self.correlated_only_1_corr, self.correlated_only_2_corr)), self.correlated_both_2_corr))
        
        if plot:
            self.plot_correlated_mz()
            
    def plot_correlated_mz(self, suptitle = False):
        fig, axs = plt.subplots(4, 1, figsize = (15, 15))

        ax = axs[0]
        ax.plot(self.correlated_all_mz, self.correlated_all_1_corr, 'o', ms = 2)
        ax.hlines(self.minimum_correlation_with_insulin_1, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'r', '--', lw = 1, label = 'Threshold - $Ins2$')
        ax.hlines(self.minimum_correlation_with_insulin_2, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'k', '--', lw = 1, label = 'Threshold - $Ins1$')
        ax.set_xlabel('m/z')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlated with $Ins1$ or $Ins2$')
        ax.set_ylim(0,1)
        ax.legend()

        ax = axs[1]
        ax.plot(self.correlated_only_1_mz, self.correlated_only_1_corr, 'o', ms = 2)
        ax.hlines(self.minimum_correlation_with_insulin_1, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'r', '--', lw = 1, label = 'Threshold - $Ins2$')
        ax.set_xlabel('m/z')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlated with only $Ins2$')
        ax.set_ylim(0,1)
        ax.legend()

        ax = axs[2]
        ax.plot(self.correlated_only_2_mz, self.correlated_only_2_corr, 'o', ms = 2)
        ax.hlines(self.minimum_correlation_with_insulin_2, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'k', '--', lw = 0.5, label = 'Threshold - $Ins1$')
        ax.set_xlabel('m/z')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlated with only $Ins1$')
        ax.set_ylim(0,1)
        ax.legend()

        ax = axs[3]
        ax.plot(self.correlated_both_mz, self.correlated_both_1_corr, 'o', ms = 2, label = 'Correlation with $Ins2$')
        ax.plot(self.correlated_both_mz, self.correlated_both_2_corr, 'o', ms = 2, label = 'Correlation with $Ins1$')
        ax.hlines(self.minimum_correlation_with_insulin_1, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'r', '--', lw = 1, label = 'Threshold - $Ins2$')
        ax.hlines(self.minimum_correlation_with_insulin_2, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'k', '--', lw = 1, label = 'Threshold - $Ins1$')
        ax.set_xlabel('m/z')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlated with both $Ins1$ and $Ins2$')
        ax.set_ylim(0,1)
        ax.legend()
        if suptitle:
            plt.suptitle('Pearson Correlation Analysis of m/z values with $Ins1$ and $Ins2$', fontsize = 16)
        plt.tight_layout()
        plt.show()
        
        
    def create_clusters(self, k = 4, plot = True, plot_insulin2 = False, suptitle = False):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(self.correlated_all_mz.reshape(-1,1))
        
        # Get cluster labels and centroids
        centroids = kmeans.cluster_centers_
        
        borders = np.array([])
        ordered_centroids = centroids[np.argsort(centroids, axis = 0).flatten()]
        for item in range(len(centroids)-1):
            border = (ordered_centroids[item + 1] + ordered_centroids[item])/2
            borders = np.append(borders, border)
            
        self.chunk_borders = borders
        self.chunk_count = k
        self.peaks_filter = np.repeat(None, k)
        
        chunks = list(range(self.chunk_count))
        for i in range(self.chunk_count):
            if i == 0:
                # chunk_peak_mz = self.peaks_1[self.peaks_1 < borders[i]]
                # chunk_peak_n = chunk_peak_mz.shape[0]
                chunk_mz = self.correlated_all_mz[self.correlated_all_mz < borders[i]]
                chunk_corr_1 = self.correlated_all_1_corr[self.correlated_all_mz < borders[i]]
                chunk_corr_2 = self.correlated_all_1_corr[self.correlated_all_mz < borders[i]]
            elif i == self.chunk_count - 1:
                # chunk_peak_mz = self.peaks_1[self.peaks_1 > borders[i-1]]
                # chunk_peak_n = chunk_peak_mz.shape[0]
                chunk_mz = self.correlated_all_mz[self.correlated_all_mz > borders[i-1]]
                chunk_corr_1 = self.correlated_all_1_corr[self.correlated_all_mz > borders[i-1]]
                chunk_corr_2 = self.correlated_all_1_corr[self.correlated_all_mz > borders[i-1]]
            else:
                # chunk_peak_mz = self.peaks_1[(self.peaks_1 > borders[i-1]) & (self.peaks_1 < borders[i])]
                # chunk_peak_n = chunk_peak_mz.shape[0]
                chunk_mz = self.correlated_all_mz[(self.correlated_all_mz > borders[i-1]) & (self.correlated_all_mz < borders[i])]
                chunk_corr_1 = self.correlated_all_1_corr[(self.correlated_all_mz > borders[i-1]) & (self.correlated_all_mz < borders[i])]
                chunk_corr_2 = self.correlated_all_1_corr[(self.correlated_all_mz > borders[i-1]) & (self.correlated_all_mz < borders[i])]
            chunks[i] = (chunk_mz, chunk_corr_1, chunk_corr_2)
        self.chunks = chunks

        if plot:
            fig, axs = plt.subplots(self.chunk_count, 1, figsize = (10, 10))
            for i in range(self.chunk_count):
                ax = axs[i]
                ax.plot(self.chunks[i][0], self.chunks[i][1], 'o', markersize = 5, label = 'Correlation - $Ins2$')
                if plot_insulin2:
                    ax.plot(self.chunks[i][0], self.chunks[i][2], 'o', markersize = 5, label = 'Correlation - $Ins1$')
                ax.set_xlabel('m/z')
                ax.set_ylabel('Correlation')
                ax.set_title(f'Chunk {i+1}')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if suptitle:
                plt.suptitle('Chunks and Peaks', fontsize = 16)
            plt.tight_layout()
            
    def find_peaks(self, height = 0, distance = None, plot = True):
        peaks_1 = []
        peaks_2 = []
        for chunk in self.chunks:
            (indices, peak_hights_dict) = sp_find_peaks(chunk[1], height = height, distance = distance)
            peaks_1_indices = np.hstack([indices.reshape(-1,1), peak_hights_dict['peak_heights'].reshape(-1,1)])
            
            (indices, peak_hights_dict) = sp_find_peaks(chunk[2], height = height, distance = distance)
            peaks_2_indices = np.hstack([indices.reshape(-1,1), peak_hights_dict['peak_heights'].reshape(-1,1)])
            peaks_1_mz = chunk[0][np.array(peaks_1_indices[:,0], dtype=int)]
            peaks_2_mz = chunk[0][np.array(peaks_2_indices[:,0], dtype=int)]
            peaks_1.append(peaks_1_mz)
            peaks_2.append(peaks_2_mz)
        self.peaks = (peaks_1, peaks_2)
        

        if plot:
            self.plot_peaks()

    def plot_peaks(self, suptitle = False):
        fig, axs = plt.subplots(self.chunk_count, 1, figsize = (10, 15))
        
        for i in range(self.chunk_count):
            ax = axs[i]
            
            ax.plot(self.peaks[0][i], np.ones(self.peaks[0][i].shape[0]), 'o', ms = 2, label = 'Peaks - $Ins2$')
            ax.plot(self.peaks[1][i], np.ones(self.peaks[1][i].shape[0]), 'o', ms = 2, label = 'Peaks - $Ins1$')
            ax.plot(self.chunks[i][0], self.chunks[i][1], 'o', ms = 2)
            ax.hlines(self.minimum_correlation_with_insulin_1, self.correlated_all_mz.min(), self.correlated_all_mz.max(), 'r', '--', lw = 1, label = 'Threshold - $Ins2$')
            ax.set_xlabel('m/z')
            ax.set_ylabel('Correlation')
            ax.set_title('Correlated m/z values and Detected Peaks')
            ax.set_xlim(self.chunks[i][0].min() - 10, self.chunks[i][0].max() + 10)
            ax.set_ylim(0,1.2)
            ax.legend()
        if suptitle:
            plt.suptitle('Pearson Correlation Analysis of m/z values and Peaks', fontsize = 16)
        plt.tight_layout()
        plt.show()
                
    def prepare_gaussian_mixture(self, n_component_max = 30, n_repeat_experiment = 10, max_iter = 100, n_init = 10, covariance_type = 'diag', plot = True, vertical_line = 10, suptitle = False):
        n_component_max = gaussian_input_check(n_component_max, self.chunk_count)
        n_repeat_experiment = gaussian_input_check(n_repeat_experiment, self.chunk_count)
        max_iter = gaussian_input_check(max_iter, self.chunk_count)
        n_init = gaussian_input_check(n_init, self.chunk_count)
        vertical_line = gaussian_input_check(vertical_line, self.chunk_count)
        
        aic_result_list = []
        bic_result_list = []
        
        for i in range(self.chunk_count):
            mz_vector = self.chunks[i][2].reshape(-1,1)
            max_component = n_component_max[i]
            if max_component >= mz_vector.shape[0]:
                max_component = mz_vector.shape[0] - 1
                print(f'Chunk {i+1} has less m/z values than the maximum number of components. Maximum number of components is set to {max_component-1}')
                
            aic_result, bic_result = gaussian_mixture_determine_n_component(mz_vector, n_component_max = max_component, n_repeat_experiment=n_repeat_experiment[i], max_iter = max_iter[i], n_init = n_init[i], covariance_type = covariance_type)
            aic_result_list.append(aic_result)
            bic_result_list.append(bic_result)
        if plot:
            plot_aic_bic(self, aic_result_list, bic_result_list, vertical_line, suptitle= suptitle)
            
        self.aic_results = aic_result_list
        self.bic_results = bic_result_list
    
    def plot_aic_bic_line(self, n_mix_gaussians, suptitle = False):
        fig, axs = plt.subplots(self.chunk_count, figsize = (15, 10))
        for i in range(self.chunk_count):
            ax = axs[i]
            ax.plot(range(1, self.aic_results[i].shape[1]+1),self.aic_results[i].mean(axis = 0), 'r', alpha = 1, lw = 1, label = 'AIC mean')
            ax.plot(range(1, self.bic_results[i].shape[1]+1),self.bic_results[i].mean(axis = 0), 'b', alpha = 1, lw = 1, label = 'BIC mean')
            ax.axvline(x = n_mix_gaussians[i], color = 'k', linestyle = '--', lw = 0.7, label = f'Vertical Line - {n_mix_gaussians[i]}')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('AIC - BIC Scores')
            ax.set_title(f'AIC - BIC Scores for Gaussian Mixture Model - Chunk {i+1}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if suptitle:
            plt.suptitle('AIC - BIC Scores for Gaussian Mixture Model - All Chunks', fontsize = 16)
        plt.tight_layout()
        plt.show()
    
    def fit_gaussian_mixture(self, n_components, max_iter = 100, n_init = 10, covariance_type = 'diag', plot = True, suptitle = False):
        if len(n_components) != self.chunk_count:
            print('n_components should have the same length as the number of chunks')
            return
        gmm_chunks = []
        for i in range(self.chunk_count):
            mz_vector = self.chunks[i][0].reshape(-1,1)
            model = GaussianMixture(n_components = n_components[i], max_iter = max_iter, n_init = n_init, covariance_type = covariance_type)
            model.fit(mz_vector)
            gmm_chunks.append(model)
        self.gmm_chunks = gmm_chunks
        
        if plot:
            self.plot_gmm_predictions(suptitle = suptitle)
        
    def plot_gmm_predictions(self, suptitle = False):
        fig, axs = plt.subplots(len(self.gmm_chunks), 1, figsize = (10, 10))
        for chunk in range(len(self.gmm_chunks)):
            model = self.gmm_chunks[chunk]
            mz_vector_to_predict = self.chunks[chunk][0].reshape(-1,1)
            mz_vector_corr = self.chunks[chunk][1].reshape(-1,1)
            labels = model.predict(mz_vector_to_predict)
            
            ax = axs[chunk]
            ax.scatter(mz_vector_to_predict, mz_vector_corr, c=labels, s = 5, cmap = 'tab20', label = 'data')
            ax.scatter(model.means_, np.ones(model.means_.shape[0]), c =range(model.n_components), cmap = 'tab20', s = 20, marker = 'x', label = 'Gaussian mean')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xlabel('m/z')    
            ax.set_ylabel('Correlation')
            ax.set_title('Gaussian Mixture Model for Chunk ' + str(chunk))
        if suptitle:
            plt.suptitle('Gaussian Mixture Model for each Chunk', fontsize = 15)
        plt.tight_layout()
        
        
    def find_peak_filter(self, chunk = 0, interp1d_kind = 'cubic', filter_window_length = 2, filter_polyorder = 1, 
                         find_peaks_height = 0.1, find_peaks_distance = None, find_peaks_threshold = None, plot = True):
    
        x = self.chunks[chunk][0]
        y = self.chunks[chunk][1]

        # Interpolate data onto a regular grid
        regular_x = np.linspace(min(x), max(x), 1000)
        interp_func = interp1d(x, y, kind=interp1d_kind)
        y_regular = interp_func(regular_x)

        # Smooth the data using the Savitzky-Golay filter
        smoothed_y = savgol_filter(y_regular, window_length=filter_window_length, polyorder=filter_polyorder)

        # Find peaks in the smoothed data
        peaks, _ = sp_find_peaks(smoothed_y, height = find_peaks_height, distance = find_peaks_distance, threshold = find_peaks_threshold)  # Adjust the height threshold as needed

        if plot:
        # Plot the original data, interpolated data, smoothed data, and detected peaks
            plt.figure(figsize = (10,4))
            plt.plot(x, y, 'o', label='Original Data', ms = 2)
            plt.plot(regular_x, smoothed_y, label='Smoothed Data')
            plt.plot(regular_x[peaks], smoothed_y[peaks], 'ro', ms = 4, label='Detected Peaks')
            plt.legend()
            plt.xlabel('m/z')
            plt.ylabel('Correlation')
            plt.title('Peak Detection with Savitzky-Golay Filter')
            plt.ylim(0,1.2)
            plt.show()
        
        self.peaks_filter[chunk] = regular_x[peaks]
        
    def fit_known_gaussian_mixture(self, plot = False, suptitle = False):
        models = []
        for chunk in range(self.chunk_count):
            X = self.chunks[chunk][0].reshape(-1,1)
            gmm = GaussianMixture(n_components=self.peaks_filter[chunk].shape[0], means_init=self.peaks_filter[chunk].reshape(-1,1))
            gmm.fit(X)
            model = [gmm, gmm.predict(X), gmm.means_]
            models.append(model)
        self.filter_models = models
        if plot:
            self.plot_known_gaussian_mixture(suptitle = suptitle)
        
    def plot_known_gaussian_mixture(self, suptitle = False):
        fig, axs = plt.subplots(self.chunk_count, figsize=(10, 10))
        for chunk in range(self.chunk_count):
            ax = axs[chunk]
            ax.scatter(self.chunks[chunk][0].reshape(-1,1), self.chunks[chunk][1].reshape(-1,1), c = self.filter_models[chunk][1], s = 5, cmap = 'tab20', label = 'data')
            ax.scatter(self.filter_models[chunk][2], np.ones(self.filter_models[chunk][2].shape), c = range(self.filter_models[chunk][0].n_components), s = 20, marker = 'x', cmap = 'tab20', label = 'Gaussian mean')
            ax.set_xlabel('m/z')
            ax.set_ylabel('Correlation')
            ax.set_title('Chunk ' + str(chunk + 1))
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if suptitle:
            plt.suptitle('Gaussian Mixture Models with Known Peaks')
        plt.tight_layout()
        plt.show()
        
    def get_all_peaks(self):
        print('Make sure you run find peak filter for each chunk!')
        all_peaks_gmm = np.vstack((self.filter_models[0][0].means_, self.filter_models[1][0].means_, self.filter_models[2][0].means_))
        self.all_candidate_peaks = all_peaks_gmm
        


def return_candidate_peaks(obj, plot=False):
    random = np.sort(np.vstack((obj.gmm_chunks[0].means_,
                                obj.gmm_chunks[1].means_,
                                obj.gmm_chunks[2].means_)).reshape(-1)).reshape(-1, 1)
    known = obj.all_candidate_peaks.reshape(-1, 1)
    if plot:
        xlim = [[2800, 4500], [5600, 6100], [11000, 12000]]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        for i in range(3):
            ax = axs[i]
            ax.plot(random, np.ones(random.shape), 'o', label='Random Peaks', ms=3)
            ax.plot(known, np.ones(known.shape) - 0.15, 'o', label='Known Peaks', ms=3)
            ax.set_xlabel('$m/z$')
            ax.set_xlim(xlim[i])
            ax.set_ylim(0.7, 1.1)
            ax.set_yticks([0.85, 1])
            ax.set_yticklabels(['Means\nRandom Peaks', 'Means\nKnown Peaks'])
            ax.set_title(f'Detected Peaks - Cluster {i+1}')
        plt.tight_layout()
        plt.show()
    return random, known
