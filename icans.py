__author__ = 'Horea Christian' #if you contribute add your name to the end of this list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from matplotlib import ticker
from chr_matplotlib import remappedColorMap
from sklearn.decomposition import PCA, FastICA, FactorAnalysis

def base(use_filter="default", data_path="~/data/faons/latest.csv", filter_name="default.csv", participant_subset="", drop_metadata=True, drop=[], clean=7, components=5, facecolor="#ffffff"):

	data_path = path.expanduser(data_path)
	filter_path = path.join(path.dirname(path.realpath(__file__)),"filters",filter_name)

	filters = pd.read_csv(filter_path, index_col=0, header=None).transpose() # transpose filters because of .csv file formatting, specify index_col to not get numbered index
	all_data = pd.read_csv(data_path)

	all_data = all_data[map(lambda y: len(set(y)) > clean,np.array(all_data))]

	# drops metadata
	if drop_metadata == True:
		all_data = all_data.drop(filters["metadata"][pd.Series.notnull(filters["metadata"])], axis=1)

	# compile list of column names to be dropped:
	drop_list = []
	for drop_item in drop:
		drop_list += list(filters[drop_item][pd.Series.notnull(filters[drop_item])])
	drop_list = list(set(drop_list)) #get unique column names (the list may contain duplicates if overlaying multiple filters)
	all_data = all_data.drop(drop_list, axis=1)

	if participant_subset == "odd":
		keep_rows = all_data.index.values[1::2]
		filtered_data = all_data.ix[keep_rows]
	elif participant_subset == "even":
		keep_rows = all_data.index.values[0::2]
		filtered_data = all_data.ix[keep_rows]
	elif participant_subset == "male":
		filtered_data = all_data[all_data['My legal gender:'] == 'Male']
	elif participant_subset == "female":
		filtered_data = all_data[all_data['My legal gender:'] == 'Female']
	else:
		filtered_data = all_data

	#convert to correct type for analysis:
	filtered_data_array = np.array(filtered_data, dtype='float64')

	filtered_data_array = filtered_data_array / 100

	pca = PCA()
	S_pca_ = pca.fit_transform(filtered_data_array)

	fa = FactorAnalysis(svd_method="lapack")
	S_fa_ = fa.fit_transform(filtered_data_array)

	ica = FastICA(n_components=components, max_iter=20000, tol=0.00001)
	S_ica_ = ica.fit_transform(filtered_data_array)  # Estimate the sources

	load = ica.mixing_

	remapped_cmap = remappedColorMap(cm.PiYG, start=(np.max(load)-abs(np.min(load)))/(2*np.max(load)), midpoint=abs(np.min(load))/(np.max(load)+abs(np.min(load))), name='shrunk')
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17.5, 5), facecolor=facecolor)
	graphic = ax.imshow(load, cmap = remapped_cmap, interpolation='none')



if __name__ == '__main__':
	base(drop=["liking", "inconclusive", "overly detailed"], components=5)
	plt.show()
