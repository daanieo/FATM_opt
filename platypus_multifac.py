#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:18:34 2020

@author: daan
"""

import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geoplot
from shapely.geometry import Polygon

from platypus import NSGAII, Problem, Real
from math import ceil

def determine_number_of_silos(nb_days,
                              nb_people):

    # Fill in food basket
    food_basket = {}
    food_basket["rice"] = 0.6
    food_basket["chickpeas"] = 0.25
    food_basket["oil"] = 0.04

    # Determine the number of silos based on a 12 tonne silo
    nb_silos = 0
    for f in food_basket:
        nb_silos += ceil(food_basket[f]*nb_days*nb_people/12000.)

    return nb_silos


# Determine costs in USD based on source as defined in report
def determine_costs(nb_days,
                    nb_people):

    # Costs subdivided in fixed, per 60 and per 12 tonnes
    costs = {'fixed_inv':74,
              '60_inv':125,
              '12_inv':5,
              'fixed_op':225,
              '60_op':74,
              '12_op':10}

    # Assign total with fixed investment and operational costs
    investment_costs = costs['fixed_inv']
    operational_costs = costs['fixed_op']

    # Computer number of silos (i.e. 12 tonnes)
    nb_silos = determine_number_of_silos(nb_days=nb_days,
                                         nb_people=nb_people)

    # Assign total with costs per 12 tonnes
    investment_costs += costs['12_inv'] * nb_silos
    operational_costs += costs['12_op'] * nb_silos

    # Assign total with costs per 60 tonnes
    investment_costs += costs['60_inv'] * ceil(nb_silos/5)
    operational_costs += costs['60_op'] * ceil(nb_silos/5)

    # Return total costs, investment and operational resp.
    return investment_costs,operational_costs

def determine_pareto_front(costs):
    """
    Downloaded from:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Retrieved on: 20 October 2020

    Searching for an algorithm finding the pareto front.
    """
    import numpy as np
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_polygon_extremes(some_polygon):

    import numpy as np
    # Set initial values
    xmin = np.inf
    xmax = 0
    ymin = np.inf
    ymax = 0

    # Extract the point values that define the perimeter of the polygon
    x, y = some_polygon.exterior.coords.xy

    # Loop through list looking for extremes
    for xval in x:
        if xval > xmax:
            xmax = xval
        elif xval < xmin:
            xmin = xval
        else:
            pass
    for yval in y:
        if yval > ymax:
            ymax = yval
        elif yval < ymin:
            ymin = yval
        else:
            pass

    # Return extreme values for x and y respectively
    return [xmin,xmax],[ymin,ymax]


def penalty_road_absence(x,
                          y,
                          gdf,
                          grid_intervals):
    # The goal is to determine whether a combination x,y corresponds with a cell with road presence

    horizontal_intervals = grid_intervals
    vertical_intervals = len(gdf)/grid_intervals

    presence = 0 # standard 0, debugged for proposals outside of the considered grid

    for h in range(horizontal_intervals):

        x_extremes = get_polygon_extremes(gdf.geometry[h*vertical_intervals])[0]

        if x_extremes[0] <= x <= x_extremes[1]:
            horizontal_coord = int((h) * vertical_intervals)
            break

    for v in range(horizontal_coord,int(horizontal_coord+vertical_intervals)):
        y_extremes = get_polygon_extremes(gdf.geometry[v])[1] # only fetch y values of polygon

        if y_extremes[0] <= y <= y_extremes[1]:
            coord = v
            presence = gdf.roads[coord]
            break

    penalty = 3000 * (1-presence)

    return penalty # 0 when there's roads, penalty when there are




def platypus_optimisation(n_fac,
                          nb_days,
                          iterations,
                          buildings_df,
                          gdf,
                          grid_intervals):

    import math

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

    # Returns value for KPIs based on platypus-generated vars (= lon,lat facility)
    def determine_KPIs(vars):

        # Container for each facility's x-y coords
        fac_x_list = list() # x coords per facility
        fac_y_list = list() # y coords per facility

        # Fill container with x-y coords for each facility
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


        # Sum total costs per facility
        for index in range(len(fac_nb)):

            penalty = penalty_road_absence(x=fac_x_list[index],y=fac_y_list[index],gdf=gdf,grid_intervals=grid_intervals)
            costs = determine_costs(nb_days=nb_days,nb_people=fac_nb[index]) # NOTE: number of people equals number of buildings
            investment_costs += costs[0] + penalty
            operational_costs += costs[1]

        # Compute mean and variance of distance
        mean_distance = np.mean(dist_b)
        #var_distance = np.var(dist_b)
        var_distance = np.max(dist_b)

        # Return results
        return [mean_distance,
                var_distance,
                investment_costs,
                operational_costs]

    # Extreme x-y coordinates for the zone02 polygon
    extr_x=[31.3165843, 31.3849562]
    extr_y=[3.5106438, 3.5476851]

    # Problem definition
    problem = Problem(2*n_fac,4) # 2 coordinates per facility, 1 KPI
    p_types = list()
    for f in range(n_fac):
        p_types.append(Real(extr_x[0],extr_x[1]))
        p_types.append(Real(extr_y[0],extr_y[1]))
    problem.types[:] = p_types
    problem.function = determine_KPIs
    problem.directions[:] = problem.MINIMIZE

    # Define algorithm and run
    algorithm = NSGAII(problem)
    algorithm.run(iterations)

    return algorithm.result


def outcome_generation (n_fac_min,
                        n_fac_max,
                        nb_days_min,
                        nb_days_max,
                        buildings_df,
                        gdf,
                        grid_intervals,
                        iterations):

    results_as_dict = {'nb_facs':list(),
                       'nb_days':list(),
                       'loc_facs':list(),
                       'mean_dist':list(),
                       'var_dist':list(),
                       'inv_costs':list(),
                       'op_costs':list()}

    metric_names = ['mean_dist','var_dist','inv_costs','op_costs']

    for n_fac in range(n_fac_min,n_fac_max+1):

        for nb_days in range(nb_days_min,nb_days_max+1):

            print("Optimisation started")
            one_result = platypus_optimisation(n_fac=n_fac,nb_days=nb_days,buildings_df=buildings_df,gdf=gdf,grid_intervals=grid_intervals,iterations=iterations)

            for r in one_result:
                results_as_dict['nb_facs'].append(n_fac)
                results_as_dict['nb_days'].append(nb_days)
                results_as_dict['loc_facs'].append(r.variables)

                for o in range(len(r.objectives)):
                    results_as_dict[metric_names[o]].append(r.objectives[o])

    results_as_df = pd.DataFrame(results_as_dict)

    results_as_df.to_csv("save_files/raw_results.csv")

    return results_as_df



def determine_efficiency(costs):

    # Returns True if left is pareto optimal to w respect to the right, else False
    def compare_two(left,right): # left optimal to right?
        score = np.zeros(len(left),dtype=bool)
        for i in range(len(left)):
            score[i]=left[i]<right[i]
        if np.any(score):
            return True
        else:
            return False

    is_efficient = np.zeros(len(costs),dtype=bool)

    for c in range(len(costs)):

        self_efficient = True

        for i in range(c):

            if is_efficient[i] == True:

                if compare_two(costs[i],costs[c]):
                    is_efficient[i]=True
                    self_efficient=compare_two(costs[c],costs[i])
                else:
                    is_efficient[i]=False

        is_efficient[c]=self_efficient

    return is_efficient
