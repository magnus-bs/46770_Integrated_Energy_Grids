#%%
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_functions as pf


#%% ----------------------------------------------------------------------------------------------
#                                 Load relevant data
### ----------------------------------------------------------------------------------------------

# Load heat load data
heat_demand=pd.read_csv("data/heat_demand.csv",sep=';')
heat_demand.index = pd.DatetimeIndex(heat_demand['utc_time'])

# Load temperature data for Basel (used to calculate COP)
df = pd.read_csv("data/temperature_Basel.csv", sep=",")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Keep data up to last hour of 2025
df_temp = df[df["timestamp"] <= "2025-12-31 23:00:00"]

#%% ----------------------------------------------------------------------------------------------
#                                 Create networks to compare
### ----------------------------------------------------------------------------------------------

# BASE NETWORK
#------------------------------
# Copy of nodal network created in main_file.py
network_base = pypsa.Network("nodal_network.nc")

# NEW NETWORK WITH LAND TRANSPORT
#------------------------------
# Create a copy of the base network to modify
#network_base.model.solver_model = None
network_multi=network_base.copy()

# Add multi-carrier components to the new network

# Add heat bus to each country
countries={"FR": "FRA", "BE": "BEL", "CH": "CHE", "IT": "ITA", "DE": "DEU"}
for country in countries:
    network_multi.add("Bus", f"{country}_heat")

# Add heat demand to each heat bus 
    network_multi.add("Load", f"{country}_heat_demand", bus=f"{country}_heat", p_set=heat_demand[countries[country]].values)    

#%% Add unlimited gas storage in each country
for country in countries:
    network_multi.add("Bus", f"{country}_gas")

# We add a gas store with energy capacity and an initial filling level much higher than the required gas consumption, 
# this way gas supply is unlimited 
for country in countries:
    network_multi.add("Store", f"{country}_gas_store", bus=f"{country}_gas", e_nom=1e9, e_initial=1e9)


# Add combined heat and power (CHP) plants to each country, with a fixed ratio between electricity 
# and heat production, and fuelled by gas from the gas store
def annuity(n, r):
    """ Calculate the annuity factor for an asset with lifetime n years and
    discount rate r """
    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n
    
annutiy_factor_CPH=annuity(n=25, r=0.07)
annulised_capital_cost_CHP=annutiy_factor_CPH*600000*(1 + 0.03) # €/MW


for country in countries:
    network_multi.add(
        "Link",
        f"{country}_CHP",
        bus0=f"{country}_gas",
        bus1=country,
        bus2=f"{country}_heat",
        p_nom=10**3, # 1 GW of fuel input
        marginal_cost=53, # €/MWh fuel cost
        capital_cost=annulised_capital_cost_CHP,
        efficiency=0.49,
        efficiency2=0.43,
        p_min_pu=0
    )


# Add heat pumps to italy and france, with a fixed COP that depends on the temperature in Basel
annutiy_factor_HP=annuity(n=20, r=0.07)
annulised_capital_cost_HP=annutiy_factor_HP*933000*(1 + 0.035) # €/MW

def cop(t_source, t_sink=55):
    delta_t = t_sink - t_source
    return 6.81 - 0.121 * delta_t + 0.00063 * delta_t**2


temp=df_temp["temperature"]


for country in ["FR", "IT"]:


    network_multi.add(
        "Link",
        f"{country}_heat_pump",
        carrier="heat pump",
        bus0=country,
        bus1=f"{country}_heat",
        efficiency=cop(temp).values,
        p_nom_extendable=True,
        capital_cost=annulised_capital_cost_HP,
        p_min_pu=0
        )  # €/MWe/a


#%% ----------------------------------------------------------------------------------------------
#                                 Optimize both networks
### ----------------------------------------------------------------------------------------------

#network_base.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})
network_multi.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})