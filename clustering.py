import numpy as np
from sklearn import mixture
from part4 import NUM_OF_PARTIES


class GMM:

    def __init__(self):
        self.model = None
        self.components = 0
        self.cvType = ''

    def fit(self, train, val):
        lowest_bic = np.infty
        n_components_range = range(2, NUM_OF_PARTIES)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(train)
                bic = gmm.bic(val)
                if bic < lowest_bic:
                    lowest_bic = bic
                    self.model = gmm
                    self.components = n_components
                    self.cvType = cv_type

    def predict(self, test):
        return self.model.predict(test)
