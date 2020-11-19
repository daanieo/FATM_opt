#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:52:10 2020

@author: daan
"""

from platypus_multifac import outcome_generation
import geopandas
import pandas as pd
import seaborn

grid_gdf = geopandas.read_file("save_files/grid_70/grid.shp")
buildings_df_zone02 = pd.read_csv("save_files/InitialRun/buildings_zone02.csv")
x_steps=70

d = {}

for i in range(20):

    raw_results = outcome_generation  (n_fac_min=6,
                                       n_fac_max=6,
                                       nb_days_min=1,
                                       nb_days_max=1,
                                      buildings_df=buildings_df_zone02,
                                       gdf=grid_gdf,
                                       grid_intervals=x_steps,
                                      iterations=50*i)
    d[i] = raw_results   

l = list()

for i in d:
    
    for j in range(len(d[i])):
        l.append(d[i].iloc[j,:])

from platypus_multifac import determine_pareto_front
import numpy as np 

# Only metric columns
raw_results_as_input = np.array(raw_results.iloc[:,-4:])
# boolean list of efficient rows
is_efficient = determine_pareto_front(raw_results_as_input)
# prepare list with to be dropped columns 
drop_index = list()
for i in range(len(is_efficient)):
    if is_efficient[i]==False:
        drop_index.append(i)

# assign pareto optimal rows to results 
raw_results_p = raw_results.drop(index=drop_index).reset_index()
raw_results_p = raw_results_p.iloc[:,2:]

seaborn.set_context( rc={"axes.labelsize":15})
pp = seaborn.pairplot(raw_results_p.iloc[:,:])
pp.fig.suptitle("Pairplot: raw, Pareto efficient results",y=1.03,size=25);

print("List has ",len(raw_results_p)," results")