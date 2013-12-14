#!/usr/bin/env python
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
__author__ = 'Horea Christian' #if you contribute add your name to the end of this list
#from import_data import get_data
import numpy as np
from os import path
from pandas import DataFrame, Series
from mdp.nodes import FANode
from pylab import *
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

#input files
filtername = 'pandas_default' #filter file
remote_data = 'http://chymera.eu/opendata/FANS/public_questionnaire_results.csv' #global data folder
local_dir = path.dirname(path.realpath(__file__)) + '/'
local_data = 'localdata/' #local data folder
local_filename = 'res2013-06-18' # survey results from said date
filter_dir= 'filters/' #location of filter files

#filepaths (if not local then global) 
if path.isfile(local_dir + local_data + local_filename):
    datapath = local_dir + local_data + local_filename + '.csv'
else: 
    datapath = remote_data
filterpath = local_dir + filter_dir + filtername + '.csv'

filters = DataFrame.from_csv(filterpath).transpose() # transpose filters because of .csv file formatting
all_data = DataFrame.from_csv(datapath)
half_data_rows = all_data.index.values[::2] # select only odd indexes (keep the other dataset half for validation)
half_data = all_data.ix[half_data_rows]
prefiltered_data = half_data.drop(filters['inconclusive'][Series.notnull(filters['inconclusive'])], axis=1) # drop inconclusive fields
prefiltered_data = prefiltered_data.drop(filters['overly detailed'][Series.notnull(filters['overly detailed'])], axis=1) # drop overly detailed fields

m_data = prefiltered_data[prefiltered_data['My legal gender:'] == 'Male'] # separate male dataset
f_data = prefiltered_data[prefiltered_data['My legal gender:'] == 'Female'] # separate female dataset

filtered_data = prefiltered_data.filter(filters['liking']) # filter fields of interest
filtered_data = filtered_data.filter(filters['passive'])

cleaned_data_df = filtered_data[map(lambda y: len(set(y)) > 7,np.array(filtered_data))] # clean respondents with less than 7 different replies
cleaned_data = np.array(cleaned_data_df, dtype='float64')
cleaned_data = cleaned_data / 100

fa = FANode(output_dim= 5, max_cycles=50000,tol=0.0000000001)
fa.train(cleaned_data)
scores = fa.execute(cleaned_data)
#~ loading = fa.get_projmatrix()
projection_matrix = fa.E_y_mtx
#~ print projection_matrix[0], scores[0]#, cleaned_data_df.columns.tolist(), np.shape(scores), np.shape(projection_matrix)


fit = r.factanal(cleaned_data, 5, rotation='promax')
load = r.loadings(fit)
load = rpyn.ri2numpy(load)
#~ load = load[:,2:]

load = r.t(load)
fig = figure(facecolor='#eeeeee',  tight_layout=True)
ax = fig.add_subplot(111)
ax.imshow(load, cmap = cm.PiYG, interpolation='none')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
ax.set_xticklabels(cleaned_data_df.columns.tolist(),fontsize=8,rotation=90)
show()

#~ fig = figure(facecolor='#eeeeee',  tight_layout=True)
#~ ax = fig.add_subplot(111)
#~ ax.imshow(projection_matrix.T, cmap = cm.PiYG, interpolation='none')
#~ ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
#~ ax.set_xticklabels(cleaned_data_df.columns.tolist(),fontsize=8,rotation=90)
#~ show()
