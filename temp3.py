#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:14:08 2020

@author: daan
"""

import math
from platypus import *
import numpy as np

from data_prep_funcs import get_polygon_extremes

import geopandas
import pandas as pd

buildings_df = pd.read_csv("save_files/buildings_all_zones_df.csv")
grid_gdf = geopandas.read_file("save_files/grid_250.shp")
grid_intervals = 250
n_fac = 12

n_iterations = 10000




def penalty_road_absence(x,
                          y,
                          grid_gdf=grid_gdf,
                          grid_intervals=grid_intervals):
    # The goal is to determine whether a combination x,y corresponds with a cell with road presence

    horizontal_intervals = grid_intervals
    vertical_intervals = len(grid_gdf)/grid_intervals

    presence = 0 # standard 0, debugged for proposals outside of the considered grid

    for h in range(horizontal_intervals):

        x_extremes = get_polygon_extremes(grid_gdf.geometry[h*vertical_intervals])[0]

        if x_extremes[0] <= x <= x_extremes[1]:
            horizontal_coord = int((h) * vertical_intervals)
            break

    for v in range(horizontal_coord,int(horizontal_coord+vertical_intervals)):
        y_extremes = get_polygon_extremes(grid_gdf.geometry[v])[1] # only fetch y values of polygon

        if y_extremes[0] <= y <= y_extremes[1]:
            coord = v
            presence = grid_gdf.roads[coord]
            break

    penalty = (1-presence) #* 3000

    return penalty # 0 when there's roads, penalty when there are


def compute_distance(lata,latb,lona,lonb):
    R = 6373

    lata = math.radians(lata)
    lona = math.radians(lona)
    latb = math.radians(latb)
    lonb = math.radians(lonb)

    dlat = lata-latb
    dlon = lona-lonb

    a = math.sin(dlat / 2)**2 + math.cos(lata) * math.cos(latb) * math.sin(dlon / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R*c

    return distance

def determine_KPIs(vars,
                    buildings_df=buildings_df):

    # Initialise and fill both container for x-y coords
    fac_x_list = list() # x coords per facility
    fac_y_list = list() # y coords per facility
    for a in range(len(vars)):
        if a%2==0:
            fac_x_list.append(vars[a])
        else:
            fac_y_list.append(vars[a])

    # Emtpy containers for ppl per facility, distance per building resp.
    fac_nb = np.zeros(n_fac)
    dist_b = np.zeros(len(buildings_df))

    # Fill in the containers
    for b in range(len(buildings_df)):

        b_x = buildings_df.x[b]
        b_y = buildings_df.y[b]
        min_distance = np.inf

        for f in range(len(fac_x_list)):
            fac_x = fac_x_list[f]
            fac_y = fac_y_list[f]

            # tmp_distance = np.sqrt( ( (b_x-fac_x)**2 + (b_y-fac_y)**2 ) )
            tmp_distance = compute_distance(b_x,fac_x,b_y,fac_y)

            if tmp_distance<min_distance:
                min_distance=tmp_distance
                closest_fac = f

            else:
                pass

        fac_nb[closest_fac] += 1
        dist_b[b] = min_distance

    # Total inv and op costs
    investment_costs = 0
    operational_costs = 0

    dev_coverage = list()

    # Sum total costs per facility
    for index in range(len(fac_nb)):

        penalty = penalty_road_absence(x=fac_x_list[index],y=fac_y_list[index])
        #costs = determine_costs(nb_days=nb_days,nb_people=fac_nb[index]) # NOTE: number of people equals number of buildings

        investment_costs += penalty
        # operational_costs += costs

        dev_coverage.append(np.max([0,(fac_nb[index] - 2500*2.587)]))

    # Compute mean and variance of distance
    mean_distance = np.mean(dist_b)
    #var_distance = np.var(dist_b)
    var_distance = np.max(dist_b)


    # Return results
    return [mean_distance,
            var_distance,
            investment_costs,
            np.max(dev_coverage)]


# Extreme x-y coordinates for the bidibidi polygon
extr_x = [31.20268, 31.5128083]
extr_y = [3.1833812, 3.5854929]

hmin = np.zeros(2*n_fac)
hmax = np.zeros(2*n_fac)

for i in range(n_fac):
    hmin[2*i] = extr_x[0]
    hmin[2*i+1] = extr_y[0]
    hmax[2*i] = extr_x[1]
    hmax[2*i+1] = extr_y[1]

algorithms = [NSGAII]#, (NSGAIII, {"divisions_outer":12})]
problem = Problem(2*n_fac,4) # 2 coordinates per facility, 4 KPI
p_types = list()
for f in range(n_fac):
    p_types.append(Real(extr_x[0],extr_x[1]))
    p_types.append(Real(extr_y[0],extr_y[1]))
problem.types[:] = p_types
problem.function = determine_KPIs
problem.directions[:] = problem.MINIMIZE

# problem.constraints[:] = [">=0",">=0","==0",">=0"]

problems = [problem]

with ProcessPoolEvaluator(10) as evaluator:
    # run the experiment
    results = experiment(algorithms, problems, nfe=20000,evaluator=evaluator,seeds=10)
    
    # calculate the hypervolume indicator
    hyp = Hypervolume(minimum=[0,0,0,0], maximum=[5,25,n_fac,10000])
    hyp_result = calculate(results, hyp)
    display(hyp_result, ndigits=3)

# DONT FORGET TO UPDATE THIS: nfe1000_seeds_algorithm_C
run_name = "nfe20000_10_NSGAII_NC"

hyp_list = list()
hyp_list.append(hyp_result['NSGAII']['Problem']['Hypervolume'])

pd.DataFrame(hyp_list).to_csv("save_files/hypervolumes_%s.csv"%run_name)


varlist = list()
oblist = list()

for j in results['NSGAII']['Problem']:
    for k in j:
        varlist.append(k.variables)
        oblist.append(k.objectives[:])
   
objectives = pd.DataFrame(oblist)
variables = pd.DataFrame(varlist)

objectives.to_csv("save_files/objectives_%s.csv"%run_name)
variables.to_csv("save_files/variables_%s.csv"%run_name)

            
import seaborn
objectives = pd.DataFrame(oblist)#[0:100])
objectives.columns = ["avg_distance","max_distance","penalty","div_coverage"]
seaborn.set_context( rc={"axes.labelsize":15})
pp = seaborn.pairplot(objectives)#raw_results.iloc[:,1:])
pp.fig.suptitle("Pairplot: %s"%run_name,y=1.03,size=25);

from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.express as px


fig = px.parallel_coordinates(objectives, 
                              #color="nb_facs", 
                              #dimensions = ["0","1","2","3"],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              #color_continuous_midpoint=4.5,
                              title="Parallel axis plot %s"%run_name)
plot(fig)

