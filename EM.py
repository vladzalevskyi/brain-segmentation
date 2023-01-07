from sklearnex import patch_sklearn
patch_sklearn()


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


class ExpectationMaximization:
    def __init__(self, n_clusters:int, init_method='kmeans'):
        """Expectation Maximization class constructor

        Args:
            n_clusters (int): Number of classes to segment images into
            init_method (str, optional): Type of parameters initialization.
                Either 'kmeans', 'segm' or 'points'. Defaults to 'kmeans'.
        """
        
        self.n_clusters = n_clusters
        self.init_method = init_method
        
        # stores training progress
        self._training_data = {'ll':[]}
    
    def segment(self, images:list, mask=None,
                tissue_atlasses:list = None,
                segmentation=None, max_iter=1000, tol=1e-3):
        """Segment a given image using EM algorithm.

        Args:
            images (list): list of np.ndarray image volumes of
                different modalities to be used for segmentation 
            mask (np.ndarray, optional): Optional boolean mask used to 
                select which of volumes voxels to segment. Defaults to None.
            tissue_atlases (list): list of np.ndarray image volumes of
                different tissues probabilistic atlases (should be registered
                to the image we want to segment) in order of [CSF, WM, GM]
            segmentation(np.ndarray, optional): Segmentation labels used for 
                initialization if init_method == 'segm'
        """
        
        # extract voxels to segment
        if mask is not None:
            images_flattened = [x[mask!=0] for x in images]
        else:
            images_flattened = [x.ravel() for x in images]
        
        # convert them to a matrix Nxm, where N is number of voxels and m is number of modalities
        X = np.stack(images_flattened).T
        X = X.reshape(X.shape[0], -1)
                
        # initialize parameters of the models
        if self.init_method == 'kmeans':
            self.init_params_kmeans(X)
        
        elif self.init_method == 'points':
            self.init_params_points(X)
        
        elif self.init_method == 'segm':
            if segmentation is None:
                raise ValueError("Provide proper segmentation for this initialization")
            mask = np.ones_like(images[0]) if mask is None else mask 
            segmentation = segmentation[mask!=0]
            self.init_params_segm(X, segmentation)
        
        else:
            raise ValueError("Not supported initialization type")
        
        # initialize atlas probabilities
        if tissue_atlasses is not None:
            if mask is not None:
                atlases_flattened = [x[mask!=0] for x in tissue_atlasses]
            else:
                atlases_flattened = [x.ravel() for x in tissue_atlasses]
            
            # convert them to a matrix Nxm, where N is number of voxels and m is number of modalities
            self.atlas_probs = np.stack(atlases_flattened).T
            self.atlas_probs = self.atlas_probs.reshape(self.atlas_probs.shape[0], -1)
        else:
            self.atlas_probs = None
        
        # when running only k-means
        if max_iter == 0:
            self.expectation_step(X)
        
        # EM loop
        for i in range(max_iter):
            
            self.expectation_step(X)
            self.maximization_step(X)
            
            # calculate metrics for progress tracking and update the progress bar
            self._training_data['ll'].append(self.log_likelihood())

            ll_diff = self.ll_difference(num_iter=1)
            print(ll_diff)
            
            if i != 0:
                if (ll_diff)<tol:
                    break
        
        # obtain final segmentation mask
        labels = np.argmax(self.W, axis=1) + 1
        res_mask = np.zeros(images[0].shape)
        res_mask[mask!=0] = labels
        
        
        # remapping cluster labels to match GT
        res_mask = self.remap_mask(images, mask, res_mask)
        
        return res_mask

    def init_params_kmeans(self, X):
        """Initialize parameters of the model by using kmeans algorithm
            It will define alpha, mus and covs base on the kmeans clusters"""
        
        kmeans = KMeans(n_clusters=self.n_clusters)
        label = kmeans.fit_predict(X)
        center = kmeans.cluster_centers_

        # initialize alpha as the proportion of points in each cluster
        self.alpha = np.ones(self.n_clusters)/self.n_clusters
        for idx, clust in enumerate(np.unique(label)):
            self.alpha[idx] = np.sum(label==clust)/label.shape[0]
        
        self.mus = []
        self.covs = []
        for clust in range(self.n_clusters):
            self.mus.append(center[clust])
            self.covs.append(np.cov(X[label==clust], rowvar=False))

    def init_params_segm(self, X, segmentation):
        """Initialize EM with tissue models

        Args:
            segmentation (np.ndarray): 3D volume with discrete segmentation
                labels.
        """
        # calculate alphas as proportion between pixel classes
        labs_count = {np.uint8(k):v for k,v in dict(zip(*np.unique(segmentation, return_counts=True))).items() if k!=0}
        self.alpha  = np.asarray([labs_count[k]/np.sum(list(labs_count.values())) for k in range(1, self.n_clusters + 1)])

        self.mus = []
        self.covs = []
        
        for clust in range(1, self.n_clusters + 1):
            self.mus.append(np.mean(X[segmentation==clust]))
            self.covs.append(np.cov(X[segmentation==clust], rowvar=False))

    def init_params_points(self, X):
        """Initialize parameters of the model by selecting random points from X"""
        self.alpha = np.ones(self.n_clusters)/self.n_clusters
        
        self.mus = [X[np.random.choice(X.shape[0])] for _ in range(self.n_clusters)]
        
        self.covs = [np.cov(X, rowvar=False) for _ in range(self.n_clusters)]
    
    def expectation_step(self, X):
        """Expectation step of EM algorithm
        Sets up membership weights for each point in X"""
        
        # get gaussian mixture model probabilities weighed by alpha
        self.gmm_probs = np.zeros((X.shape[0], self.n_clusters))
        for clust in range(self.n_clusters):
            self.gmm_probs[:, clust] = multivariate_normal.pdf(X,
                                                          self.mus[clust],
                                                          self.covs[clust])*self.alpha[clust]
        # normalize gmm_probs so they become weights
        self.W = np.zeros((X.shape[0], self.n_clusters))
        for clust in range(self.n_clusters):
            self.W[:,clust]  = self.gmm_probs[:, clust] / np.sum(self.gmm_probs, axis=1)

        # integration of atlas into EM (only if atlas is provided)
        if self.atlas_probs is not None:
            self.W = self.W * self.atlas_probs
        
        
    def maximization_step(self,X):
        """Maximization step of EM algorithm
        Recalculates parameters of the model (alpha, mu, cov)"""
        self.Nk = self.W.sum(axis=0)
        
        self.alpha = self.Nk/self.W.shape[0]
        
        self.mus = [(self.W[:, clidx].reshape(-1,1)*X).sum(axis=0)/self.Nk[clidx]\
            for clidx in range(self.n_clusters)]
        
        
        self.covs = [((self.W[:, clidx].reshape(-1,1) *  (X - self.mus[clidx])).T.dot(X - self.mus[clidx]))/self.Nk[clidx]\
            for clidx in range(self.n_clusters)]
    
    def log_likelihood(self):
        """Calculate log likelihood of the model"""
        return np.log(self.gmm_probs.sum(axis=1)).sum()
    
    def ll_difference(self, num_iter=5):
        """ Calculate difference between current and previous log likelihood.
            If num_iter is greater than 1, it will calculate average difference
            between last num_iter log likelihoods
        """
        if len(self._training_data['ll']) < num_iter + 1:
            if len(self._training_data['ll']) == 1:
                return self._training_data['ll'][-1]
            else:
                return np.abs(self._training_data['ll'][-1] - self._training_data['ll'][-2])
        
        ll_np = np.array(self._training_data['ll'])
        return np.mean(np.abs(ll_np[-num_iter:] - ll_np[-num_iter-1:-1]))
    
    @staticmethod
    def plot_segmentation(res_mask, image, mask, slice_num=24, save=False):
        """Plots segmentation results

        Args:
            res_mask (np.ndarray): 3D volume with predicted labels
            image (np.ndarray): 3D volume with original image (one modality)
            mask (np.ndarray): 3D volume with ground truth labels
            slice_num (int, optional): Slice to plot. Defaults to 24.
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        

        axs[0].imshow(image[:,:,slice_num], cmap='gray')
        axs[0].axis('off')
        axs[1].imshow(mask[:,:,slice_num], cmap='gray')
        axs[1].axis('off')
        axs[2].imshow(res_mask[:,:,slice_num], cmap='gray')
        axs[2].axis('off')
        plt.tight_layout()
        
        if save:
            plt.savefig(save)
            plt.close(fig)
        else:
            plt.show()

    def plot_training_results(self):
        """Plot training results: Log Likelihood and Dice score"""
        x = list(range(1, len(self._training_data['ll']) + 1))
        ll = self._training_data['ll']


        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        axs.plot(ll, 'g-')
        axs.set_ylabel('Log likelihood')
        axs.set_xlabel('Iterations')
        axs.set_title('Log likelihood vs iterations')
        axs.legend(['Log likelihood'])
        
        plt.show()

    @staticmethod
    def compute_dice(gt_vol: np.ndarray, pred_vol: np.ndarray, map_dict: dict={1:1, 2:2, 3:3}):
        """
        Computes dice for three tissues (CSF, GM, WM)

        Args:
            gt_vol (np.ndarray): Ground truth labels of brain volume
            pred_vol (np.ndarray): Predicted labels of brain volume
            map_dict (dict): Map of labels between GT and predicted.
                gt:pred with the gt order of  {1: 'CSF', 2: 'WM', 3: 'GM'} 

        Returns:
            dict: Dice scores for each tissue
        """
        dice_results = {}
        names = {1: 'ds_csf', 2: 'ds_gm', 3: 'ds_wm'}
        for gt, pred in map_dict.items():
            y_true = gt_vol == gt
            y_pred = pred_vol == pred
            dice_results[names[gt]] = ExpectationMaximization.single_dice_coef(y_true, y_pred)
        return dice_results
    
    @staticmethod
    def single_dice_coef(y_true: np.ndarray, y_pred_bin: np.ndarray):
        """
        Computes single class dice coef

        Args:
            y_true (np.ndarray: bool): true labels. 
            y_pred_bin (np.ndarray: bool): predicted labels

        Returns:
            int: dice similarity coefficient
        """
        intersection = np.sum(y_true * y_pred_bin)
        if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
            return 1
        return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))
    
    
    def remap_mask(self, images, gt_mask, pred_mask):
        """Remaps mask to the ground truth labels"""
        pred__tissue_means = [np.array([x[pred_mask==clust].mean() for x in images]) for clust in np.unique(pred_mask[pred_mask!=0])]
        gt_tissue_means = [np.array([x[gt_mask==clust].mean() for x in images]) for clust in np.unique(gt_mask[gt_mask!=0])]

        mapping = dict()
        for i, pred_clust in enumerate(pred__tissue_means):
            dists = [np.linalg.norm(np.array(pred_clust) - np.array(gt_clust)) for gt_clust in gt_tissue_means]
            mapping[i + 1] = np.argmin(dists) + 1

        
        # mapping: {gt_label: pred_label}
        # if repeated choose second closest cluster for repeated labels
        for not_mapped_tissue in set(mapping.keys()).difference(set(mapping.values())):
            dists = [np.linalg.norm(np.array(pred__tissue_means[not_mapped_tissue-1]) - np.array(gt_clust)) for gt_clust in gt_tissue_means]
            dists[np.argmin(dists)] = np.inf
            mapping[np.argmin(dists) + 1] = not_mapped_tissue
        
        remapped_mask = np.zeros_like(pred_mask)
        for pred, gt in mapping.items():
            remapped_mask[pred_mask == pred] = gt
        
        return remapped_mask