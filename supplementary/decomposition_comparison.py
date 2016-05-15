__author__ = "Horea Christian, Alexandre Gramfort, Gael Varoquaux"
__license__ = "BSD"

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.decomposition import PCA, FastICA, FactorAnalysis

# create data
rng = np.random.RandomState(42)
S = rng.normal(scale=0.01,size=(10000, 2))
S[0] = S[1]+S[0]/10
S[:,1][::2] *= 1.7
S[:,0][::2] /= 1.7
S[:,1][1::2] /= 1.7
S[:,0][1::2] *= 1.7
X=deepcopy(S)
X[:,1] = X[:,0]/-2+X[:,1]

# run analysis
pca = PCA()
S_pca_ = pca.fit_transform(X)

fa = FactorAnalysis(svd_method="lapack")
S_fa_ = fa.fit_transform(X)

ica = FastICA(max_iter=20000, tol=0.00001)
S_ica_ = ica.fit_transform(X)  # Estimate the sources

# plotting
def plot_samples(S, axis_list=None):
	plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10, color='steelblue', alpha=0.5)
	if axis_list is not None:
		colors = ['yellow', 'orange', 'red']
		for color, axis in zip(colors, axis_list):
			axis /= axis.std()
			x_axis, y_axis = axis
			# Trick to get legend to work
			plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
			plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6, color=color)

	plt.hlines(0, -3, 3)
	plt.vlines(0, -3, 3)
	plt.xlim(-3, 3)
	plt.ylim(-3, 3)
	plt.xlabel('x')
	plt.ylabel('y')

axis_list = [pca.components_.T, fa.components_.T, ica.mixing_]
plt.subplot(2, 2, 1)
plot_samples(X / np.std(X), axis_list=axis_list)
legend = plt.legend(['PCA','FA', 'ICA'], loc='upper right')
legend.set_zorder(100)
plt.title('Observations')

plt.subplot(2, 2, 2)
plot_samples(S_pca_ / np.std(S_pca_))
plt.title('PCA recovered signals')

plt.subplot(2, 2, 3)
plot_samples(S_fa_ / np.std(S_fa_))
plt.title('FA recovered signals')

plt.subplot(2, 2, 4)
plot_samples(S_ica_ / np.std(S_ica_))
plt.title('ICA recovered signals')

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
plt.show()
