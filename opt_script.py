if __name__ == "__main__":

    import os 

    
    import pandas as pd
    import geopandas
    from opt_funcs import optimise

    buildings_df = pd.read_csv("save_files/buildings_all_zones_df.csv")
    grid_gdf = geopandas.read_file("save_files/grid_250.shp")
    grid_intervals = 250

    res = optimise(n_fac=12,
                   n_iterations=20000,
                   buildings_df=buildings_df,
                   grid_gdf=grid_gdf,
                   grid_intervals=grid_intervals)

    vardict = {}
    obdict = {}
    for r in range(len(res.result)):
        vardict[r] = res.result[r].variables
        obdict[r] = res.result[r].objectives[:]
        
    pd.DataFrame(obdict).to_csv("save_files/ob_20000.csv")
    pd.DataFrame(vardict).to_csv("save_files/var_20000.csv")


    os.system("shutdown")
