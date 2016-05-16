import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, FastICA, FactorAnalysis

rng = np.random.RandomState(42)
s = rng.normal(scale=0.01,size=(4,1000))
S = np.ones((3,1000))
S[0] = s[0]
S[1] = s[1]
S[2] = s[0]+s[1]

pca = PCA()
S_pca_ = pca.fit_transform(S.T)

fa = FactorAnalysis(svd_method="lapack")
S_fa_ = fa.fit_transform(S.T)

ica = FastICA(max_iter=20000, tol=0.00001)
S_ica_ = ica.fit_transform(S.T)  # Estimate the sources

def plot_3d(data, ax, axis_list=None):
	data /= np.std(data)
	ax.scatter(data[0] ,data[1], data[2] , s=2, marker='o', zorder=10, color='steelblue', alpha=0.5)
	# plt.hlines(0, -4, 4)
	# plt.vlines(0, -4, 4)
	ax.set_xlim(-4, 4)
	ax.set_ylim(-4, 4)
	ax.set_zlim(-4, 4)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
		label.set_fontsize(6)
	if axis_list is not None:
		colors = ['yellow', 'orange', 'red']
		for color, axis in zip(colors, axis_list):
			axis /= axis.std()
			for subaxis in axis.T:
				x_axis, y_axis, z_axis = subaxis
				ax.plot([0,x_axis],[0,y_axis],[0,z_axis], linewidth=2, color=color)

axis_list = [pca.components_.T, fa.components_.T, ica.mixing_]
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(221, projection='3d')
plot_3d(S, ax, axis_list)

ax = fig.add_subplot(222, projection='3d')
plot_3d(S, ax)

ax = fig.add_subplot(223, projection='3d')
plot_3d(S, ax)

ax = fig.add_subplot(224, projection='3d')
plot_3d(S, ax)

plt.show()
