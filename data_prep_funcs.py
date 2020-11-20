#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:18:34 2020

@author: daan
"""

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

# Construct a GeoDataFrame of the accessibility grid
def make_grid_gdf(extr_x,
                      extr_y,
                      x_steps,
                      roads_shp):
    from shapely.geometry import Polygon
    import geopandas as gp
    import numpy as np
    # import math


    def make_grid_polygons(extr_x,
                           extr_y,
                           x_steps):

        # Function creating a polygon for every grid cell
        def make_polygon(xar,
                         yar):
            minx = xar[0]
            maxx = xar[1]
            miny = yar[0]
            maxy = yar[1]
            return Polygon([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]])

        xlist,ylist = make_grid(extr_x=extr_x,extr_y=extr_y,x_steps=x_steps) # construct grid

        list_with_polygons = list() # emtpy container

        # Fill the empty container with polygons
        for i in range(len(xlist)-1):
            for j in range(len(ylist)-1):
                xar = [xlist[i],xlist[i+1]]
                yar = [ylist[j],ylist[j+1]]
                list_with_polygons.append(make_polygon(xar=xar,yar=yar))

        return list_with_polygons


    # Get list with polygons, representing grid cells
    list_with_polygons = make_grid_polygons(extr_x=extr_x,
                                            extr_y=extr_y,
                                            x_steps=x_steps)


    list_of_gdfs = list()
    gdf_as_dict = {}

    gdf_as_dict['idnr'] = list()
    gdf_as_dict['roads'] = list()
    gdf_as_dict['geometry'] = list()

    gdf_as_dict['x_coord'] = list()
    gdf_as_dict['y_coord'] = list()


    for i in range(len(list_with_polygons)):
        d={}
        d['idnr'] = i
        d['roads'] = False
        d['geometry'] = [list_with_polygons[i]]

        gdf_as_dict['idnr'].append(i)
        gdf_as_dict['roads'].append(False)
        gdf_as_dict['geometry'].append(list_with_polygons[i])

        xtmp,ytmp=get_polygon_extremes(list_with_polygons[i])

        gdf_as_dict['x_coord'].append(np.sum(xtmp)/2)
        gdf_as_dict['y_coord'].append(np.sum(ytmp)/2)

        list_of_gdfs.append(gp.GeoDataFrame(d))

    # Count the number of building polygons for every grid cell
    for index in range(len(list_of_gdfs)):

        print("We are at ",index," Out of ", len(list_of_gdfs))

        # Spatial join of buildings/roads and grid cell
        sjoined_roads = gp.sjoin(roads_shp,list_of_gdfs[index]);

        # Add nr of buildings to previously created dictionary
        gdf_as_dict['roads'][index] = len(sjoined_roads)>0

    gdf = gp.GeoDataFrame(gdf_as_dict); # Create geodataframe from dict

    return gdf

# Function serving as input for plotting the grid and construction of the cell polygons
def make_grid(extr_x,
              extr_y,
              x_steps):

    # Determine step size
    stapgrootte = (extr_x[1] - extr_x[0]) / x_steps

    # Empty lists for storage of intermediate x and y coordinates
    x_list = list()
    y_list = list()

    # Fill list with x values
    for i in range(x_steps+1):
        x_list.append(extr_x[0] + stapgrootte * i)

    # Some logic for creating intermediate y values
    go = True
    count = 0

    # Adding stapgrootte to ymin, till ymax has been reached
    while go:
        theone = extr_y[0] + count*stapgrootte
        count+=1
        if theone > extr_y[1]:
            go = False
            break
        y_list.append(theone)

    # Return the list with intermediate values for x and y respectively
    return x_list,y_list

def make_road_heatmap(grid_gdf,
                      roads_shp,
                      extr_x,
                      extr_y,
                      x_steps):
    import geoplot
    import geopandas
    import math
    import matplotlib.pyplot as plt
    def compute_cell_size(extr_x=extr_x,extr_y=extr_y,number_of_steps=x_steps):

        # approximate radius of earth in km
        R = 6373.0

        lat1 = math.radians(extr_y[0])
        lon1 = math.radians(extr_x[0])
        lat2 = math.radians(extr_y[0])
        lon2 = math.radians(extr_x[0]+(extr_x[1]-extr_x[0])/number_of_steps)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) *math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        print("Cell size is ", distance," km")

    compute_cell_size()
    dims = (10, 6) # define dimensions tuple
    fig,ax = plt.subplots(1,1,figsize=dims) # Define figurex axis for subplots and dimensions

    plt.xlim([extr_x[0],extr_x[1]]) # set xlim
    plt.ylim([extr_y[0],extr_y[1]]) # set ylim
    plt.title('Heatmap for cells with road connection') # set title

    geoplot.choropleth(grid_gdf,ax=ax,hue=grid_gdf['roads'],legend=True,cmap='RdYlGn') # draw heatmap
    roads_shp.plot(ax=ax,color='yellow') # draw roads


# Save a road shapefile; preparing convenient in case of very big road shapefile
def roads_within_polygon(address_of_road_shp, # File address
                         polygon): # The to be joined polygon

    # Import functionality
    import geopandas as gp

    # Load road shape file
    road_shapefile = gp.read_file(address_of_road_shp)

    # Spatially join roads over the given polygon
    roads_in_camp_within = gp.sjoin(road_shapefile,polygon,op='within')
    roads_in_camp_intersect = gp.sjoin(road_shapefile,polygon,op='intersects')

    return roads_in_camp_intersect,roads_in_camp_within



# Save a shapefile of buildings to csv containing points instead of polygons
def buildings_shp_to_df(address_of_shp):

    # Import functionality
    import geopandas as gp
    import pandas as pd
    import numpy as np

    # Read shapefile containing building shapefiles from address
    buildings=gp.read_file(address_of_shp)

    # Emtpy containers
    df_as_dict = {}
    id_nr_list = list()
    x_list = list()
    y_list = list()

    # Loop over shapefile with buildings
    for i in range(len(buildings)):
        x_extr,y_extr = get_polygon_extremes(buildings.geometry[i]) # Get extreme edges
        x_list.append( np.sum(x_extr) / 2. ) # Get horizontal centre
        y_list.append( np.sum(y_extr) / 2. ) # Get vertical centre
        id_nr_list.append(i) # Add to container

    # Add to dict container
    df_as_dict["id_nr"]=id_nr_list
    df_as_dict["x"]=x_list
    df_as_dict["y"]=y_list

    df = pd.DataFrame(df_as_dict)

    return df
