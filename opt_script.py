if __name__ == "__main__":

    import os 

    

    from platypus_multifac import outcome_generation
    import geopandas
    import pandas as pd

    grid_gdf = geopandas.read_file("save_files/grid_70/grid.shp")
    buildings_df_zone02 = pd.read_csv("save_files/InitialRun/buildings_zone02.csv")
    x_steps=70

    raw_results = outcome_generation  (n_fac_min=1,
                                       n_fac_max=15,
                                       nb_days_min=1,
                                       nb_days_max=1,
                                      buildings_df=buildings_df_zone02,
                                       gdf=grid_gdf,
                                       grid_intervals=x_steps,
                                      iterations=1000)
    os.system("shutdown")
