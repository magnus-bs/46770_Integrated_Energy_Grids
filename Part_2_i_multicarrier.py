#%%
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_functions as pf
import importlib
importlib.reload(pf)

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

# Add unlimited gas storage in each country
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

# calculate annulised captial cost for CHP    
annutiy_factor_CPH=annuity(n=25, r=0.07)
annulised_capital_cost_CHP=annutiy_factor_CPH*600000*(1 + 0.03) # €/MW


# Add CHP plants to each country
for country in countries:
    network_multi.add(
        "Link",
        f"{country}_CHP",
        bus0=f"{country}_gas",
        bus1=country,
        bus2=f"{country}_heat",
        p_nom_extendable=True,
        marginal_cost=21.6,
        capital_cost=annulised_capital_cost_CHP,
        efficiency=0.49,
        efficiency2=0.43,
        p_min_pu=0
    )

# Add heat pumps to italy and france, with a fixed COP that depends on the temperature in Basel
annutiy_factor_HP=annuity(n=20, r=0.07)
annulised_capital_cost_HP=annutiy_factor_HP*933000*(1 + 0.035) # €/MW

# Define COP function
def cop(t_source, t_sink=55):
    delta_t = t_sink - t_source
    return 6.81 - 0.121 * delta_t + 0.00063 * delta_t**2

# Get temperature data
temp=df_temp["temperature"]

# Add heat pumps to france and italy
for country in ["FR", "IT", "CH"]:

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

# Calculate annulised capital cost for the boiler
annutiy_factor_boiler=annuity(n=20, r=0.07)
annulised_capital_cost_boiler=annutiy_factor_CPH*63000*(1 + 0.015) # €/MW

# Add boilers to each country, fuelled by gas from the gas store
for country in countries:
    network_multi.add("Link", f"{country}_boiler",
        bus0=f"{country}_gas", bus1=f"{country}_heat",
        p_nom_extendable=True,
        efficiency=0.9,
        marginal_cost=21.6, # €/th_MWh fuel cost,
        capital_cost=annulised_capital_cost_boiler)
    
#%% ----------------------------------------------------------------------------------------------
#                                 Optimize both networks
### ----------------------------------------------------------------------------------------------

network_base.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})
network_multi.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

#%% ----------------------------------------------------------------------------------------------
#                                 Results and analysis
### ----------------------------------------------------------------------------------------------

# -------------- Compare total system cost------------------------------
print("\n=== Total System Cost (M€) ===")
print(f'Basecase: {round(network_base.objective/1e6,2)}')

print(f'Multi-carrier system: {round(network_multi.objective /1e6,2)}')

# Print how big objective is compared to base case
print(f"Multi-carrier system is {round((network_multi.objective - network_base.objective)/network_base.objective*100,2)}% more expensive than the base case")

#  -------------- Optimal Capacity and generation mix--------------------
print("\n=== Optimal Capacity (MW) ===")

# Check generator capacities
print(round(network_multi.generators.p_nom_opt, 2)) # Check generators

# Check link capacities 
print(round(network_multi.links.p_nom_opt.groupby(level=0).sum(), 2)) # Check links

print("\n=== Optimal Generation (GWh) ===")

# Check generation from generators
print(round(network_multi.generators_t.p.sum()/1000, 2)) # Check generators

# Check generation from links
for country in countries:
    power = -network_multi.links_t.p1[f"{country}_CHP"].sum() / 1000
    heat = -network_multi.links_t.p2[f"{country}_CHP"].sum() / 1000
    gas = network_multi.links_t.p0[f"{country}_CHP"].sum() / 1000

    print(f"\n{country} CHP outputs:")
    print(f"  Electricity generation: {power:.2f} GWh")
    print(f"  Heat generation:        {heat:.2f} GWh")
    print(f"  Gas input:              {gas:.2f} GWh")

for country in ["FR", "IT","CH"]:
    print(f"{country} heat pump generation (GWh): {-network_multi.links_t.p1[f'{country}_heat_pump'].sum()/1000:.2f}")

# Plot dispatch  for each country in different subplots
print("\n=== Plot dispatch ===")
pf.plot_energy_mix_multi(
    network=network_multi,
    countries=countries,
    country_order=["FR", "DE", "CH", "IT", "BE"]
)

#%%-------------------- Demand analysis -----------------------------------
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)  # in MWh
df_elec.index = pd.to_datetime(df_elec.index)

# Find heat demand for each country and electricity demand for each country in GWh
# and the ratio between the two
for country in countries:
    print(f"\n{country} electricity demand (GWh): {round(df_elec[countries[country]].sum()/1e3, 2)}")
    print(f"{country} heat demand (GWh): {round(heat_demand[countries[country]].sum()/1e3, 2)}")
    print(f"{country} heat to electricity ratio: {round(heat_demand[countries[country]].sum()/df_elec[countries[country]].sum(), 2)}")   

# Total electricity demand and heat demand and ratio
total_heat_demand = heat_demand[countries.values()].sum().sum() / 1e3  # in GWh
total_electricity_demand = df_elec[countries.values()].sum().sum() / 1e3  # in GWh
print(f"\nTotal heat demand (GWh): {round(total_heat_demand, 2)}")
print(f"Total electricity demand (GWh): {round(total_electricity_demand, 2)}")
print(f"Total heat to electricity ratio: {round(total_heat_demand/total_electricity_demand, 2)}")
print(f"The sum of heat and electricity demand is {round(total_heat_demand + total_electricity_demand, 2)} GWh  ")


# Investigation of heat and electricity demand of all countries throughout the year
for country in countries:
    df_plot = pd.concat([heat_demand[countries[country]], df_elec[countries[country]]], axis=1)
    df_plot.columns = ["Heat demand", "Electricity demand"]
    df_plot.plot(figsize=(10, 4), ylabel="Demand (MWh)", title=f"{country} heat and electricity demand")    


