import geopandas
from data_prep_funcs import get_polygon_extremes

gdf = geopandas.read_file("/home/daan/Documents/FATM_opt/save_files/grid_250.shp")
roads = geopandas.read_file("/home/daan/Documents/input_files/roads/roads_all_zones_within.shp")

bidibidi = geopandas.read_file("/home/daan/Documents/input_files/adm_boundaries/camp_polygon.shp")

extr_x,extr_y = get_polygon_extremes(bidibidi.geometry[0])
print("Extreme x values are ",extr_x,"\nExtreme y values are ",extr_y)

x_steps=250



def make_road_heatmap(grid_gdf=gdf,
                      roads_shp=roads,
                      extr_x=extr_x,
                      extr_y=extr_y,
                      x_steps=x_steps):
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

    a4_dims = (11.7, 8.27)
    fig,ax = plt.subplots(1,1,figsize=a4_dims) # Define figure and axis

    plt.xlim([extr_x[0],extr_x[1]]) # set xlim
    plt.ylim([extr_y[0],extr_y[1]]) # set ylim
    plt.title('Heatmap for cells with road connection') # set title

    geoplot.choropleth(grid_gdf,ax=ax,hue=grid_gdf['roads'],legend=True,cmap='RdYlGn') # draw heatmap
    roads_shp.plot(ax=ax,color='yellow') # draw roads