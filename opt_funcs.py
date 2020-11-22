# Returns value for KPIs based on platypus-generated vars (= lon,lat facility)
def optimise (n_fac,
                n_iterations,
                buildings_df,
                grid_gdf,
                grid_intervals):

    import math
    from platypus import NSGAII, Problem, Real
    import numpy as np

    from data_prep_funcs import get_polygon_extremes


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

            dev_coverage.append(np.abs(fac_nb[index] - 2500*2.587))

        # Compute mean and variance of distance
        mean_distance = np.mean(dist_b)
        #var_distance = np.var(dist_b)
        var_distance = np.max(dist_b)

        print("Call")

        # Return results
        return [mean_distance,
                var_distance,
                investment_costs,
                np.max(dev_coverage)]


    # Extreme x-y coordinates for the bidibidi polygon
    extr_x = [31.20268, 31.5128083]
    extr_y = [3.1833812, 3.5854929]


    # Problem definition
    problem = Problem(2*n_fac,4) # 2 coordinates per facility, 4 KPI
    p_types = list()
    for f in range(n_fac):
        p_types.append(Real(extr_x[0],extr_x[1]))
        p_types.append(Real(extr_y[0],extr_y[1]))
    problem.types[:] = p_types
    problem.function = determine_KPIs
    problem.directions[:] = problem.MINIMIZE

    # Define algorithm and run
    algorithm = NSGAII(problem)
    algorithm.run(n_iterations)

    return algorithm
