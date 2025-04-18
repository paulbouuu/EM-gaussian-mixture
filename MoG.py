import os
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse


class MoG:

    def __init__(self, k, domain = [-5, 5]):
        """
        Initialize the MoG class with parameters

        Parameters:
            - k (int): number of clusters
            - domain (list): domain of the data
        """
        self.k = k
        self.domain = domain
        self.dim = 2

        self.iteration = 0

        self.mu, self.sigma, self.pi = self._initialize_parameters()

        self.images_dir = "images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        else:
            # remove all images in the directory
            for file in os.listdir(self.images_dir):
                if file.endswith(".png"):
                    os.remove(os.path.join(self.images_dir, file))
                    
    
    def _initialize_parameters(self):
        """
        Initialize the parameters of the MoG model
        """
        # initialize means randomly within the domain
        self.mu = np.random.rand(self.k, self.dim) * (self.domain[1] - self.domain[0]) + self.domain[0]

        # initialize covariance matrices to identity matrices
        self.sigma = np.zeros((self.k, self.dim, self.dim))

        for i in range(self.k):
            self.sigma[i] = np.eye(self.dim)

        self.pi = np.ones(self.k) / self.k

        return self.mu, self.sigma, self.pi

    def init_EM_algorithm(self, X):
        """
        Initialize the EM algorithm with the given data.

        Parameters:
            - X (np.ndarray): input data for the EM algorithm
        """
        self.data = X
        self.n = X.shape[0]

        # initialize responsibilities
        self.responsibilities = np.zeros((self.n, self.k))
    
    def E_step(self):
        """
        Perform the E-step of the EM algorithm. It computes the responsibilities for each data point.
        """
        for i in range(self.k):
            self.responsibilities[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        
        # normalize responsibilities
        self.responsibilities /= np.sum(self.responsibilities, axis=1, keepdims=True)
    
    def M_step(self):
        """
        Perform the M-step of the EM algorithm. It updates the parameters based on the current responsibilities.
        """

        self.iteration += 1
        
        # update weights
        self.pi = np.sum(self.responsibilities, axis=0) / self.n

        # update means
        for i in range(self.k):
            self.mu[i] = np.sum(self.responsibilities[:, i][:, np.newaxis] * self.data, axis=0) / np.sum(self.responsibilities[:, i])

        # update covariances
        for i in range(self.k):
            diff = self.data - self.mu[i]
            self.sigma[i] = np.dot((self.responsibilities[:, i][:, np.newaxis] * diff).T, diff) / np.sum(self.responsibilities[:, i])

    def log_likelihood(self):
        """
        Computes the log-likelihood of the data given the current parameters.
        """
        log_likelihood = 0

        for point in self.data:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(point, mean=self.mu[i], cov=self.sigma[i])
            log_likelihood += np.log(tot)
            
        return log_likelihood
    
    def plot_MoG(self, plotting = True, n_std = 2):
        """
        Plot the MoG with distinct colored ellipse borders and no fill.

        Inputs:
            - plotting (bool): whether to display the plot interactively
            - n_std (int): number of standard deviations for the ellipse
        """
        fig, ax = plt.subplots()
        # scatter data points
        ax.scatter(self.data[:, 0], self.data[:, 1], s=5, alpha=0.5)

        # colormap for distinct colors
        cmap = plt.get_cmap('tab10')

        for i in range(self.k):
            mean = self.mu[i]
            cov = self.sigma[i]
            eigvals, eigvecs = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigvals)

            color = cmap(i % cmap.N)
            ell = Ellipse(mean, width, height, angle=angle,
                          edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(ell)

        ax.set_title(f"Iteration {self.iteration}")
        image_path = os.path.join(self.images_dir, f"MoG_{self.iteration}.png")
        fig.savefig(image_path)
        if plotting:
            plt.show()
        plt.close(fig)

