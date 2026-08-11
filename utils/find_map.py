from utils.basic_utils import load_data, make_image, find_nearest_idx
from utils.method_utils import generate_custom_cmap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
import skfuzzy as fuzz


from scipy.ndimage import convolve, gaussian_filter, binary_dilation
import cv2

class msi_map:
    def __init__(self, path):
        '''
        obj.correlated_insulin has 4 columns: m/z value, correlation, correlated with which insulin (0 both, 1 insulin1, 2 insulin2) and expressed in nonislets (boolean)
        '''
        dataset = path[-2:]
        self.data_all, self.residual, self.mz_vector, self.row2grid = load_data(path, dataset = dataset)
        self.data_all[self.data_all < 0] = 0
        
        path_index = path + '/cluster_indexes.npy'

        with open(path_index, 'rb') as f:
            self.cluster_labels = np.load(f)
        self.cluster_labels_map = make_image(self.row2grid, self.cluster_labels)

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
        
            
    def apply_edge_detection(self, mz = None, kernel_x = None, kernel_y = None, gaussian_smooth_sigma_dilation = 1.0,  dilation_threshold = 1, dilation_circle_size = 1,
                         gaussian_smooth_sigma_cv2 = 1, islet_alpha = 0.5, plot = True, plot_hist = True):
        if mz == None:
            self.m_z_index_for_clustering = self.insulin_1_index
            image = self.data[self.m_z_index_for_clustering]
        else:
            self.m_z_index_for_clustering = self.mz_to_index(mz)
            image = self.data[self.m_z_index_for_clustering]
            
        if kernel_x == None:
            kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        if kernel_y == None:
            kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        
        ### method dilation
        smoothed_image = gaussian_filter(image, sigma = gaussian_smooth_sigma_dilation)  
        
        edges_x = convolve(smoothed_image, kernel_x)
        edges_y = convolve(smoothed_image, kernel_y)
        
        edges_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        edge_map = edges_magnitude > dilation_threshold
        # dilation
        circle_map_bin = binary_dilation(edge_map, structure=np.ones((dilation_circle_size, dilation_circle_size)))
        
        ### method cv2
        image_gaussian = gaussian_filter(image, sigma = gaussian_smooth_sigma_cv2)   
        # Convert edge map to binary image
        _, binary_edge = cv2.threshold(image_gaussian.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        # # Find contours in the binary edge map
        contours, _ = cv2.findContours(binary_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty area map
        circle_map_cv = np.zeros_like(binary_edge,dtype=np.uint8)
        cv2.drawContours(circle_map_cv, contours, -1, 255, thickness=cv2.FILLED)

        if plot:
            fig, axs = plt.subplots(2,4, figsize = (10,8))
            plt.grid(False)
            ax = axs[0,0]
            plt.grid(False)
            ax.imshow(image, cmap='hot')
            ax.set_title('Original Image')
            
            ax = axs[0,1]
            plt.grid(False)
            ax.imshow(image, cmap='hot')
            plt.grid(False)
            ax.imshow(edges_magnitude, alpha = islet_alpha, cmap='gray')
            ax.set_title('Edges Magnitude')
            
            ax = axs[0,2]
            plt.grid(False)
            ax.imshow(image, cmap='hot')
            plt.grid(False)
            ax.imshow(circle_map_bin, cmap = 'gray', alpha = islet_alpha)
            ax.set_title('Islets of Langerhans')
            
            ax = axs[0,3]
            plt.grid(False)
            ax.imshow(image, cmap='hot')
            plt.grid(False)
            ax.imshow(circle_map_cv, alpha = islet_alpha, cmap = 'gray')
            ax.set_title('Islets of Langerhans - CV2')
            
            ax = axs[1,0]
            plt.grid(False)
            ax.imshow(image, cmap='hot')
            plt.grid(False)
            ax.set_title('Original Image')
            
            ax = axs[1,1]
            plt.grid(False)
            ax.imshow(edges_magnitude, cmap='hot')
            ax.set_title('Edges Magnitude')
            
            ax = axs[1,2]
            plt.grid(False)
            ax.imshow(circle_map_bin, cmap = 'gray')
            ax.set_title('Islets of Langerhans')
            
            ax = axs[1,3]
            plt.grid(False)
            ax.imshow(circle_map_cv, cmap = 'gray')
            ax.set_title('Islets of Langerhans - CV2')
        
            plt.tight_layout()
            plt.show()
        
        if plot_hist:
        
            fig, axs = plt.subplots(2,3, figsize = (10,8))

            image_copy_cv_ins = image.copy()
            image_copy_cv_non = image.copy()
            image_copy_bin_ins = image.copy()
            image_copy_bin_non = image.copy()
            
            circle_map_cv_bool = circle_map_cv.astype(bool)
            image_copy_cv_ins[~circle_map_cv_bool] = 0 
            image_copy_cv_non[circle_map_cv_bool] = 0
            
            circle_map_bin_bool = circle_map_bin.astype(bool)
            image_copy_bin_ins[~circle_map_bin_bool] = 0 
            image_copy_bin_non[circle_map_bin_bool] = 0
            
            ax = axs[1,0]
            plt.grid(False)
            ax.imshow(image_copy_cv_non, cmap = 'hot')
            ax.set_title('CV2 - Non-Islet Insulin Distribution')
            
            ax = axs[1,1]
            plt.grid(True)
            hist1_ins = ax.hist(image_copy_cv_ins.reshape(-1), log = True, width = 100, bins = 10, range = (0,1000))
            ax.set_xlabel('Intensity of Insulin')
            ax.set_ylabel('Frequency')
            ax.set_title('CV2 - Islet')
            ax.set_ylim(1, 10**5)

            ax = axs[1,2]
            hist1_non = ax.hist(image_copy_cv_non.reshape(-1), log = True, width = 100, bins = 10, range = (0,1000))
            ax.set_xlabel('Intensity of Insulin')
            ax.set_ylabel('Frequency')
            ax.set_title('CV2 - Non-Islet')
            ax.set_ylim(1, 10**5)

            ax = axs[0,0]
            plt.grid(False)
            ax.imshow(image_copy_bin_non, cmap = 'hot')
            ax.set_title('Dilation - Non-Islet Insulin Distribution')

            ax = axs[0,1]
            plt.grid(True)
            hist2_ins = ax.hist(image_copy_bin_ins.reshape(-1), log = True, width = 100, bins = 10, range = (0,1000))
            ax.set_xlabel('Intensity of Insulin')
            ax.set_ylabel('Frequency')
            ax.set_title('Dilation - Islet')
            ax.set_ylim(1, 10**5)

            ax = axs[0,2]
            hist2_non = ax.hist(image_copy_bin_non.reshape(-1), log = True, width = 100, bins = 10, range = (0,1000))
            ax.set_xlabel('Intensity of Insulin')
            ax.set_ylabel('Frequency')
            ax.set_title('Dilation - Non-Islet')
            ax.set_ylim(1, 10**5)

            plt.tight_layout()
        
        self.edges_magnitude = edges_magnitude
        self.circle_map_bin = circle_map_bin
        self.circle_map_cv = circle_map_cv
        
 
    def determine_area_of_interest(self, x_range = (140,170), y_range = (80,110), m_z = None):
        if m_z == None:
            m_z_index = self.m_z_index_for_clustering
        else:
            m_z_index = self.mz_index(m_z)
        data_try = self.data.copy()
        data_try[:,~self.circle_map_bin.astype(bool)] = 0   
        plt.grid(False)
        plt.imshow(data_try[m_z_index, x_range[0]:x_range[1],y_range[0]:y_range[1]], cmap = 'hot')

        data_try = data_try[:,x_range[0]:x_range[1],y_range[0]:y_range[1]]
        self.data_to_cluster = data_try
        self.data_to_cluster_x_range = x_range
        self.data_to_cluster_y_range = y_range
    
    def prepare_for_clustering(self):
        original_shape = self.data_to_cluster.shape
        reshaped_array = np.zeros((original_shape[1]*original_shape[2], original_shape[0] + 2))

        for mz in range(original_shape[0]):
            for x in range(original_shape[1]):
                for y in range(original_shape[2]):
                    reshaped_array[x*(original_shape[2]) + y, 0] = x
                    reshaped_array[x*(original_shape[2]) + y, 1] = y       
                    reshaped_array[x*(original_shape[2]) + y, mz+2] = self.data_to_cluster[mz, x, y]
                    
        self.data_to_cluster_reshaped = reshaped_array
    
    def kmeans_determine_n_component(self, max_component = 10):        
        distortions = []
        K_range = range(1, max_component)
        for k in K_range:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data_to_cluster_reshaped)
            distortions.append(kmeans.inertia_)

        plt.plot(K_range, distortions, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method for Optimal k')
        plt.show()
    
        
    def fit_cluster_k_means(self, num_clusters = 4, mz_to_plot = None, plot = True):

        # Initialize the KMeans model
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        # Fit the model to your data
        kmeans.fit(self.data_to_cluster_reshaped)

        # Get the cluster labels
        labels = kmeans.labels_

        # Reshape the labels to the shape of the original image
        x_delta = self.data_to_cluster_x_range[1] - self.data_to_cluster_x_range[0]
        y_delta = self.data_to_cluster_y_range[1] - self.data_to_cluster_y_range[0]
        
        clustered_image = labels.reshape(x_delta, y_delta)

        if plot:
            colors, custom_cmap = generate_custom_cmap(num_clusters)
            plt.figure(figsize=(10, 6))
            plt.grid(False)
            plt.imshow(clustered_image, cmap=custom_cmap, alpha = 0.7) 

            legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                                    markerfacecolor=colors[i], markersize=10) for i in range(num_clusters)]

            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5))
            plt.title('Clustering with K-Means')
            plt.show()
            
            if mz_to_plot is None:
                mz_to_plot = self.insulin_1_mz
                mz_index_to_plot = self.m_z_index_for_clustering
            else:
                mz_index_to_plot = self.mz_index(mz_to_plot)
            
            plt.figure(figsize=(10, 4))
            plt.grid(True)
            for i in range(num_clusters):
                histt = plt.hist(self.data_to_cluster[mz_index_to_plot, clustered_image == i], bins = 10, alpha = 0.7, log = True, color=colors[i], label='Cluster ' + str(i), range = (0,1000))
            plt.legend()
            plt.xlabel(f'Intensity of {mz_to_plot} m/z')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of intensity of $Ins2$ ({mz_to_plot} m/z) in each cluster')
            plt.show()

        self.data_to_cluster_clustered_kmeans = clustered_image
        
    
    def fuzzy_k_determine_n_component(self, fuzziness_param = 2, max_component = 10):
        # Generate sample data
        data = self.data_to_cluster_reshaped.copy()

        # Calculate FCM for different numbers of clusters
        fpcs = []
        for n_clusters in range(2, max_component + 1):
            cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                data.T, n_clusters, fuzziness_param, error=0.005, maxiter=1000, init=None
            )
            fpcs.append(fpc)

        # Plot the FPC (Fuzzy Partition Coefficient) for each number of clusters
        plt.figure()
        plt.plot(range(2, max_component + 1), fpcs)
        plt.xlabel("Number of clusters")
        plt.ylabel("FPC")
        plt.title("Fuzzy Partition Coefficient vs. Number of Clusters")
        plt.show()

    def fit_cluster_fuzzy_k(self, num_clusters = 4, fuzziness_param = 2, plot = True, mz_to_plot = None):

        # Generate sample data
        data = self.data_to_cluster_reshaped.copy()

        # Fuzzy C-Means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, num_clusters, fuzziness_param, error=0.005, maxiter=1000, init=None
        )

        # Membership matrix u contains the degree of membership of each data point to each cluster
        # Cluster centers are stored in cntr

        # Extracting cluster labels for each data point
        labels = np.argmax(u, axis=0)
        clustered_image = labels.reshape(self.data_to_cluster_x_range[1]-self.data_to_cluster_x_range[0],
                                         self.data_to_cluster_y_range[1]-self.data_to_cluster_y_range[0])
        
        if plot:
            colors, custom_cmap = generate_custom_cmap(num_clusters)
            plt.figure(figsize=(10, 6))
            plt.grid(False)
            plt.imshow(clustered_image, cmap=custom_cmap, alpha = 0.7) 

            legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                                    markerfacecolor=colors[i], markersize=10) for i in range(num_clusters)]

            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5))
            plt.title('Clustering with Fuzzy K')
            plt.show()        
    
            plt.figure(figsize=(10, 4))
            plt.grid(True)
            if mz_to_plot is None:
                mz_to_plot = self.insulin_1_mz
                mz_index_to_plot = self.m_z_index_for_clustering
            else:
                mz_index_to_plot = self.mz_index(mz_to_plot)
                
            for i in range(num_clusters):
                histt = plt.hist(self.data_to_cluster[mz_index_to_plot, clustered_image == i], bins = 10, alpha = 0.7, log = True, color=colors[i], label='Cluster ' + str(i), range = (0,1000))
            plt.legend()
            plt.xlabel(f'Intensity of {mz_to_plot} m/z')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of intensity of $Ins2$ ({mz_to_plot} m/z) in each cluster')
            plt.show()
            
        self.data_to_cluster_clustered_fuzzyk = clustered_image
    
        
    def fit_cluster_hdbscan(self, min_cluster_size = 5, plot = True, mz_to_plot = None):
        hdbscan_obj = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        hdbscan_obj.fit(self.data_to_cluster_reshaped)
        hdbscan_labels = hdbscan_obj.labels_
        num_clusters = np.max(hdbscan_labels) + 1
        hdbscan_labels = hdbscan_labels.reshape(self.data_to_cluster_x_range[1]-self.data_to_cluster_x_range[0], 
                                                self.data_to_cluster_y_range[1]-self.data_to_cluster_y_range[0])

        if plot:
            colors, custom_cmap = generate_custom_cmap(num_clusters)
            plt.figure(figsize = (10,6))
            plt.grid(False)
            plt.imshow(hdbscan_labels, cmap=custom_cmap)
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                                                markerfacecolor=colors[i], markersize=10) for i in range(num_clusters)]

            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5))
            plt.title('Clustering with HDBSCAN')
            plt.show()
                        
            if mz_to_plot is None:
                mz_to_plot = self.insulin_1_mz
                mz_index_to_plot = self.m_z_index_for_clustering
            else:
                mz_index_to_plot = self.mz_index(mz_to_plot)
                
            plt.figure(figsize = (10,4))
            plt.grid(True)
            for i in range(num_clusters):
                histt = plt.hist(self.data_to_cluster[mz_index_to_plot, hdbscan_labels == i], bins = 10, alpha = 0.7, log = True, color=colors[i], label='Cluster ' + str(i), range = (0,1000))
            plt.legend()
            plt.xlabel(f'Intensity of {mz_to_plot} m/z')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of intensity of $Ins2$ ({mz_to_plot} m/z) in each cluster')
            plt.show()
            
        self.data_to_cluster_clustered_hdbscan = hdbscan_labels
        
        
    