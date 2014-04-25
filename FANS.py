#!/usr/bin/env python
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
__author__ = 'Horea Christian' #if you contribute add your name to the end of this list
#from import_data import get_data
import numpy as np
from os import path
from pandas import DataFrame, Series
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from matplotlib import ticker
from rpy2.robjects import r, numpy2ri
from chr_helpers import get_config_file
from chr_matplotlib import remappedColorMap
numpy2ri.activate()

def fa(source=False, use_filter="default", data_file="latest", participant_subset="", drop_metadata=True, drop=[], clean=7, factors=5, facecolor="#ffffff"):
    #gets config file:
    config = get_config_file(localpath=path.dirname(path.realpath(__file__))+'/')
	    
    #IMPORT VARIABLES
    if not source:
	    source = config.get('Source', 'source')
    data_path = config.get('Addresses', source)
    filter_dir = config.get('Paths', "filter_dir")
    filter_name = config.get("Filters", use_filter)
    #END IMPORT VARIABLES
    
    filter_path = path.dirname(path.realpath(__file__)) + '/' + filter_dir + filter_name + '.csv'
    
    filters = DataFrame.from_csv(filter_path, header=None).transpose() # transpose filters because of .csv file formatting
    all_data = DataFrame.from_csv(data_path + data_file + ".csv")
    all_data = all_data.reset_index(level=0)
    #~ print filters["metadata"]
    
    #clean data of respondents who only ckeck extreme answers: 
    all_data = all_data[map(lambda y: len(set(y)) > clean,np.array(all_data))]

    if drop_metadata == True:
        # drops metadata
        all_data = all_data.drop(filters["metadata"][Series.notnull(filters["metadata"])], axis=1)
    
    drop_list = []
    for drop_item in drop:
        # compile list of column names to be dropped:
        drop_list += list(filters[drop_item][Series.notnull(filters[drop_item])])
    #get unique column names (the list may contain duplicates if overlaying multiple filters):
    drop_list = list(set(drop_list))
    
    all_data = all_data.drop(drop_list, axis=1)
    
    if participant_subset == "odd":
        # selects only odd indexes (keep the other dataset half for validation)
        keep_rows = all_data.index.values[1::2]
        filtered_data = all_data.ix[keep_rows]
    elif participant_subset == "even":
        # selects only even indexes (keep the other dataset half for validation)
        keep_rows = all_data.index.values[0::2] 
        filtered_data = all_data.ix[keep_rows]
    elif participant_subset == "male":
        # selects only male participants
        filtered_data = all_data[all_data['My legal gender:'] == 'Male']
    elif participant_subset == "female":
        # selects only female participants
        filtered_data = all_data[all_data['My legal gender:'] == 'Female']
    else:
        filtered_data = all_data

    #convert to correct type for analysis:
    filtered_data_array = np.array(filtered_data, dtype='float64')
   
    filtered_data_array = filtered_data_array / 100
    
    fit = r.factanal(filtered_data_array, factors, rotation='promax')
    load = r.loadings(fit)
    load = numpy2ri.ri2numpy(load)
    
    load = r.t(load)

    remapped_cmap = remappedColorMap(cm.PiYG, start=(np.max(load)-abs(np.min(load)))/(2*np.max(load)), midpoint=abs(np.min(load))/(np.max(load)+abs(np.min(load))), name='shrunk')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17.5, 5), facecolor=facecolor)  
    graphic = ax.imshow(load, cmap = remapped_cmap, interpolation='none')
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax.set_xticklabels([0]+filtered_data.columns.tolist(),fontsize=8,rotation=90)
    ax.set_yticklabels(np.arange(factors+1))
    ax.set_ylabel('Factors')
    ax.set_title("Question Loadings on Factors")

    #Recolor plot spines:
    for spine_side in ["bottom", "top", "left", "right"]:
        ax.spines[spine_side].set_color("#777777")
    
    #Remove ticks:
    plt.tick_params(axis='both', which='both', left="off", right="off", bottom='off', top='off')
    
    divider = make_axes_locatable(ax)
    #calculate width for cbar so that it is equal to the question column width:
    cbar_width = str(100/np.shape(load)[1])+ "%"
    cax = divider.append_axes("right", size=cbar_width, pad=0.05)
    cbar = colorbar(graphic, cax=cax, drawedges=True)
    
    #Limit the number of ticks:
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()

    #Align ticklabels so that negative values are not misaligned (meaning right align):
    for t in cbar.ax.get_yticklabels():
        t.set_horizontalalignment('right')   
        t.set_x(0.045*(np.shape(load)[1]+6))

    #Tweak color bar borders
    cbar.outline.set_color("#666666")
    cbar.dividers.set_linewidth(0)
    
if __name__ == '__main__':
    fa(facecolor='#eeeeee', drop=["liking", "inconclusive", "overly detailed"], factors=5)
    show()

