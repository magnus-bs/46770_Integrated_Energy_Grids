#%%
import pandas as pd
import os
if os.getlogin() == "magnu": # to fix one member's problems with pypsa's proj_path
    import pyproj
    proj_path = r"C:\Users\magnu\anaconda3\envs\E2Flex\Library\share\proj"
    pyproj.datadir.set_data_dir(proj_path)
import pypsa
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plot_functions as pf



#%% ----------------------------------------------------------------------------------------------
#                                  STEP 1: Problem Definition
### ----------------------------------------------------------------------------------------------

# Load data:
df_wind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_wind.index = pd.to_datetime(df_wind.index, utc=True)  # parse as UTC first

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index, utc=True)


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define consistent colors for technologies
tech_colors = {'demand': 'black', 'onshorewind': 'blue', 'solar': 'orange', 'OCGT': 'red','nuclear': 'brown'}

def annuity(n, r):
    """ Calculate the annuity factor for an asset with lifetime n years and
    discount rate r """
    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

# Create network
network = pypsa.Network()
hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
network.set_snapshots(hours_in_2015.values)
network.add("Bus", "FR", y=46.2, x=2.2, v_nom=400, carrier="AC")

# Load electricity demand data
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)  # in MWh
df_elec.index = pd.to_datetime(df_elec.index)
country = 'FRA'

# Add load to the bus
network.add("Load", "load", bus="FR", p_set=df_elec[country].values)

# Add carriers
network.add("Carrier", "gas", co2_emissions=0.19, overwrite=True)  # in t_CO2/MWh_th
network.add("Carrier", "nuclear", overwrite=True)
network.add("Carrier", "onshorewind", overwrite=True)
network.add("Carrier", "solar", overwrite=True)

# Add onshore wind generator
CF_wind = df_wind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]

capital_cost_onshorewind = annuity(30, 0.07) * 910000 * (1 + 0.033)  # in €/MW
network.add("Generator", "onshorewind", bus="FR", p_nom_extendable=True,
            carrier="onshorewind", capital_cost=capital_cost_onshorewind, marginal_cost=0,
            p_max_pu=CF_wind.values, overwrite=True)

# Add solar PV generator
CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]

capital_cost_solar = annuity(25, 0.07) * 425000 * (1 + 0.03)  # in €/MW
network.add("Generator", "solar", bus="FR", p_nom_extendable=True,
            carrier="solar", capital_cost=capital_cost_solar, marginal_cost=0,
            p_max_pu=CF_solar.values, overwrite=True)

# Add OCGT generator
capital_cost_OCGT = annuity(25, 0.07) * 560000 * (1 + 0.033) # in €/MW
fuel_cost = 21.6  # in €/MWh_th # set multiplier for storage case?
efficiency = 0.39
marginal_cost_OCGT = fuel_cost / efficiency  # in €/MWh_el
network.add("Generator", "OCGT", bus="FR", p_nom_extendable=True,
            carrier="gas", capital_cost=capital_cost_OCGT, marginal_cost=marginal_cost_OCGT,
            overwrite=True)

# Add nuclear generator
average_demand = df_elec[country].mean()  # in MW
capital_cost_nuclear = annuity(60, 0.07) * 3460 * 1000 * (1 + 0.03)
network.add("Generator", "nuclear", bus="FR", p_nom_extendable=True,
            carrier="nuclear", capital_cost=capital_cost_nuclear, marginal_cost=6,
            p_nom_max=average_demand*0.7, p_max_pu=1.0, p_min_pu=0.5, ramp_limit_up=0.15, ramp_limit_down=0.15,
            overwrite=True)

# Optimize without CO2 limit
network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

print("Optimization Results (No CO2 Limit):")
print(f"Total cost: {network.objective / 1000000:.2f} million €")
print(f"Levelized cost: {network.objective / (network.loads_t.p.values.sum()):.2f} €/MWh")
print("Optimal capacities (MW):")
for gen in network.generators.index:
    print(f"{gen}: {float(network.generators.p_nom_opt[gen]):.2f}")

# ----------------------------------------------------------------------------------------------
#                                 STEP 1: Visualisation of Results
### ----------------------------------------------------------------------------------------------
import importlib
import plot_functions as pf
importlib.reload(pf)

# Weekly Disptach (summer/winter)
pf.weekly_dispatch_plot(network, tech_colors, start_day = 0)
pf.weekly_dispatch_plot(network, tech_colors, start_day = 24*30*6)

colors = [tech_colors['onshorewind'], tech_colors['solar'], tech_colors['OCGT'], tech_colors['nuclear']]
labels = ['Onshore Wind', 'Solar', 'Gas (OCGT)', 'Nuclear']

# Energy Mix
pf.energy_mix_piechart(network, colors, labels, full_year = True, dpi = 300)
    
# Load Duration Curves
pf.duration_curves(network, tech_colors, figsize = (10,4), dpi = 400)








#%% ----------------------------------------------------------------------------------------------
#                                 STEP 2: Interannual Variability
### ----------------------------------------------------------------------------------------------

# Fixed demand across weather years
demand = df_elec[country].values  # will be reused for all weather years

# Years considered
years = range(2010, 2018)

# Store results:
results = []
cost_results = []

for year in years:
    print(f"Running optimization for {year}...")

    # Filter year
    CF_wind = df_wind.loc[f'{year}-01-01':f'{year}-12-31', country]
    CF_solar = df_solar.loc[f'{year}-01-01':f'{year}-12-31', country]

    # Drop Feb 29 if present (leap year)
    CF_wind = CF_wind[~((CF_wind.index.month == 2) & (CF_wind.index.day == 29))]
    CF_solar = CF_solar[~((CF_solar.index.month == 2) & (CF_solar.index.day == 29))]

    # Convert to values
    CF_wind = CF_wind.values
    CF_solar = CF_solar.values

    # Create network (Same as earlier, therefore less comments)
    network_year = pypsa.Network()
    network_year.set_snapshots(pd.RangeIndex(len(demand)))
    network_year.add("Bus", "FR", p_set=demand)
    network_year.add("Load", "load", bus="FR", p_set=demand)
    network_year.add("Carrier", "gas", co2_emissions=0.19, overwrite=True)
    network_year.add("Carrier", "nuclear", overwrite=True)
    network_year.add("Carrier", "onshorewind", overwrite=True)
    network_year.add("Carrier", "solar", overwrite=True)
    network_year.add("Generator", "onshorewind", bus="FR", p_nom_extendable=True,
                carrier="onshorewind", capital_cost=capital_cost_onshorewind, marginal_cost=0,
                p_max_pu=CF_wind, overwrite=True)
    network_year.add("Generator", "solar", bus="FR", p_nom_extendable=True,
                carrier="solar", capital_cost=capital_cost_solar, marginal_cost=0,
                p_max_pu=CF_solar, overwrite=True)
    network_year.add("Generator", "OCGT", bus="FR", p_nom_extendable=True,
                carrier="gas", capital_cost=capital_cost_OCGT, marginal_cost=marginal_cost_OCGT,
                overwrite=True)
    avg_demand = df_elec[country].mean()
    network_year.add("Generator", "nuclear", bus="FR", p_nom_extendable=True,
                carrier="nuclear", capital_cost=capital_cost_nuclear, marginal_cost=6,
                p_nom_max=avg_demand*0.7, p_max_pu=1.0, p_min_pu=0.5,
                ramp_limit_up=0.15, ramp_limit_down=0.15, overwrite=True)
    network_year.optimize(solver_name='gurobi', solver_options={"OutputFlag":0})
    
    # Store results
    for gen in network_year.generators.index:
        results.append({
            'year': year,
            'generator': gen,
            'p_nom': float(network_year.generators.p_nom_opt[gen]),
            'dispatch_std': network_year.generators_t.p[gen].std(),
            'total_dispatch': network_year.generators_t.p[gen].sum()

        })

    # Store cost results
    total_cost = network_year.objective  # in €
    total_energy = network_year.loads_t.p.sum()  # total MWh demand served
    lcoe = total_cost / total_energy  # €/MWh

    cost_results.append({
        'year': year,
        'total_cost_M€': total_cost / 1e6,
        'LCoE_€/MWh': lcoe
    })

df_costs = pd.DataFrame(cost_results)
df_results = pd.DataFrame(results)

# --------------
# VISUALISATIONS:
# --------------

pf.annual_capacities_and_dispatch(df_results, tech_colors)

largest_difference = df_costs['total_cost_M€'].max() / df_costs['total_cost_M€'].min() * 100 - 100
print("The largest percentage difference between worst and best year is "+str(round(largest_difference,2))+"%")

pf.interannual_var_boxplots(df_results)

# Plot total cost over years
plt.figure(figsize=(10,6), dpi = 300)
plt.plot(df_costs['year'], df_costs['total_cost_M€'], marker='o', linestyle='-', color='tab:blue')
plt.xlabel('Year')
plt.ylabel('Total System Cost (M€)')
plt.title('Total System Cost per Weather Year')
plt.grid(alpha=0.3)
#plt.ylim(0,df_costs['total_cost_M€'].max())
plt.show()







#%% ----------------------------------------------------------------------------------------------
#                                 STEP 3: ADDING STORAGE
### ----------------------------------------------------------------------------------------------

# NETWORK SETTINGS:
network.model.solver_model = None
network_storage = network.copy()

# Add storage unit
lifetime = 80
capital_cost_hydro = annuity(lifetime, 0.07) * 1994 * 10**3 * 0.7 # reduced by 30%
fixed_o_m = 16.46 * 10**3  # EUR/MW/yr
network_storage.add("StorageUnit", "Pumped Hydro", bus="FR", p_nom_extendable=True,
            max_hours=30, efficiency_store=0.95, efficiency_dispatch=0.85,
            capital_cost=capital_cost_hydro + fixed_o_m, marginal_cost=0, overwrite=True)

print("\nStorage unit added: Pumped Hydro")
network_storage.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

print("\nOptimization Results (With Pumped Hydro):")
print(f"Total cost: {network_storage.objective / 1e6:.2f} million €")
print(f"Levelized cost: {network_storage.objective / float(network.loads_t.p.sum().sum()):.2f} €/MWh")

# 2. Check optimal storage capacity
p_nom_storage = network_storage.storage_units.p_nom_opt["Pumped Hydro"]
print(f"Optimal Pumped Hydro Capacity: {p_nom_storage:.2f} MW")

# 3. Plot Dispatch with Storage (First week of the last year)
start_day = 24 * 30 * 6 + 24 * 15
pf.weekly_dispatch_plot(network_storage, tech_colors, start_day, storage = True, figsize=(20,8))

pf.energy_mix_piechart(network_storage, colors, labels, full_year = True)

# 4. Plot State of Charge (How full the "battery" is)
pf.weekly_soc_plot(network_storage, start_day)



#%% ----------------------------
# COMPARISON
# ----------------------------

# ── 1. Cost comparison ────────────────────────────────────────────────────────
total_demand = float(network.loads_t.p.sum().sum())

costs = {
    'Without storage': network.objective,
    'With storage':    network_storage.objective
}
lcoe = {k: v / total_demand for k, v in costs.items()}

print("=== Cost Comparison ===")
for label in costs:
    print(f"{label}: {costs[label]/1e6:.2f} M€  |  LCOE: {lcoe[label]:.2f} €/MWh")
print(f"Cost reduction: {(costs['Without storage'] - costs['With storage'])/1e6:.2f} M€ "
      f"({(1 - costs['With storage']/costs['Without storage'])*100:.1f}%)")



# ── 2. Optimal capacities comparison ─────────────────────────────────────────
cap_base    = network.generators.p_nom_opt
cap_storage = network_storage.generators.p_nom_opt

df_cap = pd.DataFrame({
    'Without storage (MW)': cap_base,
    'With storage (MW)':    cap_storage
}).round(0)
print("\n=== Optimal Capacities ===")
print(df_cap)


# ── 3. Electricity mix comparison (pie charts side by side) ───────────────────
gens = ["onshorewind", "solar", "OCGT", "nuclear"]
labels = ["Onshore wind", "Solar", "OCGT", "Nuclear"]
colors = [tech_colors[g] for g in gens]

pf.plot_energy_mix_comparison(network, network_storage, gens, labels, colors)

# ── 4. Annual dispatch comparison (stacked bar) ───────────────────────────────
pf.plot_dispatch_comparison(network, network_storage, gens, labels, colors)

# ── 5. Dispatch time series comparison (first week) ──────────────────────────
pf.plot_dispatch_timeseries(network, network_storage, gens, labels, colors, hours = 24*7)

# ── 6. Storage state of charge (full year) ───────────────────────────────────
pf.plot_storage(network_storage)



#%% 
# ----------------------- Test: Imposing CO2 Limit 

network.model.solver_model = None

network_co2 = network.copy()

# Add CO2 limit
co2_limit = 1000000  # tonCO2
network_co2.add("GlobalConstraint", "co2_limit", type="primary_energy",
            carrier_attribute="co2_emissions", sense="<=", constant=co2_limit)

# Optimize with CO2 limit
network_co2.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

print("\nOptimization Results (With CO2 Limit):")
print(f"Total cost: {network_co2.objective / 1000000:.2f} million €")
print(f"Levelized cost: {network_co2.objective / float(network_co2.loads_t.p.sum().sum()):.2f} €/MWh")
print("Optimal capacities (MW):")
for gen in network_co2.generators.index:
    print(f"{gen}: {float(network_co2.generators.p_nom_opt[gen]):.2f}")


# ── 5. Dispatch time series comparison (first week) ──────────────────────────
pf.weekly_dispatch_plot(network_co2, tech_colors, 0, storage = False, figsize = (15,5), dpi = 300)

# Pie chart for energy mix with CO2 limit
pf.energy_mix_piechart(network_co2, colors, labels, full_year = True, dpi = 300)






#%% ----------------------------------------------------------------------------------------------
#                                 STEP 4: ADDING INTERCONNECTIONS
### ----------------------------------------------------------------------------------------------

network.model.solver_model = None

network_nodes = network.copy()
# Add neighboring countries as buses
network_nodes.add("Bus", "DE", y=51.0, x=10.0, v_nom=400, carrier="AC")
network_nodes.add("Bus", "CH", y=46.8, x=8.3, v_nom=400, carrier="AC")
network_nodes.add("Bus", "IT", y=43.0, x=12.5, v_nom=400, carrier="AC")
network_nodes.add("Bus", "BE", y=50.8, x=4.4, v_nom=400, carrier="AC")

# Add load for neighboring countries
network_nodes.add("Load", "DE_load", bus="DE", p_set=df_elec["DEU"].values)
network_nodes.add("Load", "CH_load", bus="CH", p_set=df_elec["CHE"].values)
network_nodes.add("Load", "IT_load", bus="IT", p_set=df_elec["ITA"].values)
network_nodes.add("Load", "BE_load", bus="BE", p_set=df_elec["BEL"].values)

country_map = {"DE": "DEU", "CH": "CHE", "IT": "ITA", "BE": "BEL"}

for bus, code in country_map.items():
    CF_wind_n  = df_wind.loc['2015-01-01':'2015-12-31', code]
    CF_solar_n = df_solar.loc['2015-01-01':'2015-12-31', code]

    # Drop leap day if present
    CF_wind_n  = CF_wind_n[~((CF_wind_n.index.month == 2) & (CF_wind_n.index.day == 29))]
    CF_solar_n = CF_solar_n[~((CF_solar_n.index.month == 2) & (CF_solar_n.index.day == 29))]

    CF_wind_n  = CF_wind_n.values
    CF_solar_n = CF_solar_n.values

    network_nodes.add("Generator", f"{bus}_wind", bus=bus,
                p_nom_extendable=True, carrier="onshorewind",
                capital_cost=capital_cost_onshorewind, marginal_cost=0,
                p_max_pu=CF_wind_n)

    network_nodes.add("Generator", f"{bus}_solar", bus=bus,
                p_nom_extendable=True, carrier="solar",
                capital_cost=capital_cost_solar, marginal_cost=0,
                p_max_pu=CF_solar_n)

    network_nodes.add("Generator", f"{bus}_OCGT", bus=bus,
                p_nom_extendable=True, carrier="gas",
                capital_cost=capital_cost_OCGT,
                marginal_cost=marginal_cost_OCGT)

network_nodes.add("Carrier", "hydro", overwrite=True)

# Switzerland: hydropower (run-of-river()
capex_hydro        = 2600 * 1000                          # €/MW
fixed_om_hydro     = 0.025 * capex_hydro                  # 2.5% of CAPEX/year
capital_cost_hydro = annuity(80, 0.07) * capex_hydro + fixed_om_hydro

# Swiss hydro average CF ~0.45 (roughly consistent with IEA annual stats)
network_nodes.add("Generator", "CH_hydro", bus="CH",
            p_nom_extendable=True, carrier="hydro",
            capital_cost=capital_cost_hydro,
            marginal_cost=0,
            p_max_pu=1,         # constant — simple and defensible
            p_nom_max=df_elec["CHE"].mean()*0.65)

network_nodes.add("Generator", "CH_nuclear", bus="CH",
            p_nom_extendable=True, carrier="nuclear",
            capital_cost=capital_cost_nuclear,
            marginal_cost=6,
            p_nom_max=df_elec["CHE"].mean()*0.35,  # Currently 30% of electricity mix
            p_max_pu=1.0,
            p_min_pu=0.5,
            ramp_limit_up=0.15,
            ramp_limit_down=0.15)

# Belgium: nuclear
network_nodes.add("Generator", "BE_nuclear", bus="BE",
            p_nom_extendable=True, carrier="nuclear",
            capital_cost=capital_cost_nuclear,
            marginal_cost=6,
            p_nom_max=df_elec["BEL"].mean()*0.45,  # Currently 40% of electricity mix
            p_max_pu=1.0,
            p_min_pu=0.5,
            ramp_limit_up=0.15,
            ramp_limit_down=0.15)

# France connections
network_nodes.add("Line", "FR-CH", bus0="FR", bus1="CH", s_nom=3200, x=0.1, r=0)
network_nodes.add("Line", "FR-DE", bus0="FR", bus1="DE", s_nom=3000, x=0.1, r=0)
network_nodes.add("Line", "FR-IT", bus0="FR", bus1="IT", s_nom=4000, x=0.1, r=0)
network_nodes.add("Line", "FR-BE", bus0="FR", bus1="BE", s_nom=2000, x=0.1, r=0)

# extra lines to create a cycles
network_nodes.add("Line", "CH-IT", bus0="CH", bus1="IT", s_nom=4200, x=0.1, r=0)

# Run optimization
network_nodes.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

# Power flows
print(network_nodes.lines_t.p0)  # power flow from bus0 to bus1

# Generator dispatch
print(network_nodes.generators_t.p)

# Prices (dual of nodal balance)
print(network_nodes.buses_t.marginal_price)



#%% ----------------------------------------------------------------------------------------------
#                                 VISUALISATIONS
### ----------------------------------------------------------------------------------------------

# ------------------------------------- I --------------------------------------
# Stacked bar: total annual generation per technology per country

gen_by_bus = {}
for gen in network_nodes.generators.index:
    bus = network_nodes.generators.loc[gen, 'bus']
    carrier = network_nodes.generators.loc[gen, 'carrier']
    total = float(network_nodes.generators_t.p[gen].sum())
    gen_by_bus.setdefault(bus, {})
    gen_by_bus[bus][carrier] = gen_by_bus[bus].get(carrier, 0) + total

df_gen_bus = pd.DataFrame(gen_by_bus).T.fillna(0) / 1e6  # TWh

# Installed capacity per country and technology
cap_by_bus = {}
for gen in network_nodes.generators.index:
    bus = network_nodes.generators.loc[gen, 'bus']
    carrier = network_nodes.generators.loc[gen, 'carrier']
    cap = float(network_nodes.generators.p_nom_opt[gen])
    cap_by_bus.setdefault(bus, {})
    cap_by_bus[bus][carrier] = cap_by_bus[bus].get(carrier, 0) + cap

df_cap_bus = pd.DataFrame(cap_by_bus).T.fillna(0) / 1000  # GW

df_demand = pd.Series({
    "FR": df_elec["FRA"].sum()/10**6,
    "DE": df_elec["DEU"].sum()/10**6,
    "CH": df_elec["CHE"].sum()/10**6,
    "IT": df_elec["ITA"].sum()/10**6,
    "BE": df_elec["BEL"].sum()/10**6,
})

carrier_colors = {
    "onshorewind": "#4C78A8",  # muted blue
    "solar": "#F2A541",        # soft orange
    "gas": "#E45756",          # soft red
    "nuclear": "#9D755D",      # brown
    "hydro": "#72B7B2"         # teal
}

importlib.reload(pf)
pf.gen_cap_mix_stacked(df_gen_bus, df_cap_bus, df_demand, carrier_colors)




#%%
# ------------------------------------- II --------------------------------------
# Exports/Imports per country

print("Export Illustrations")
importlib.reload(pf)

pf.avg_annual_net_export_bar_plot(network_nodes)
pf.flow_matrix_heatmap(network_nodes)



#%%
# ------------------------------------- II --------------------------------------
# Line Congestion

print("Line Congestion Illustrations")

# Line loading (how congested each line is)
line_loading = network_nodes.lines_t.p0.abs() / network_nodes.lines.s_nom * 100  # in %
line_loading.mean().plot(kind='bar', figsize=(10,5), title='Average Line Loading (%)')
plt.ylabel('Loading (%)')
plt.axhline(100, color='red', linestyle='--', label='Capacity limit')
plt.legend()
plt.show()

# Flow duration curve — how often each line is congested
fig, ax = plt.subplots(figsize=(10, 5), dpi = 300)
for line in network_nodes.lines.index:
    loading = network_nodes.lines_t.p0[line].abs() / network_nodes.lines.s_nom[line] * 100
    sorted_loading = np.sort(loading.values)[::-1]
    ax.plot(sorted_loading, label=line)
ax.axhline(100, color='red', linestyle='--', label='Capacity limit')
ax.set_xlabel('Hours per year (sorted)')
ax.set_ylabel('Line loading (%)')
#ax.set_title('Flow Duration Curve per Line')
ax.legend()
plt.show()




#%%
# ------------------------------------- II --------------------------------------
# Prices

# Average marginal price per bus
avg_prices = network_nodes.buses_t.marginal_price.mean()
print(avg_prices)

# Price over time per bus
network_nodes.buses_t.marginal_price.plot(figsize=(12,5), title='Nodal Marginal Prices Over Time')
plt.ylabel('Price (€/MWh)')
plt.show()

#%%
# Define period to look at
start = '2015-01-19'
end   = '2015-01-21'  # first week of January

prices_period = network_nodes.buses_t.marginal_price.loc[start:end]

prices_period.plot(figsize=(12, 5), title=f'Nodal Marginal Prices ({start} to {end})')
plt.ylabel('Price (€/MWh)')
plt.show()





#%% ----------------------------------------------------------------------------------------------
#                                       Step E: First Time Step Information
### ----------------------------------------------------------------------------------------------

# Imbalances in each node in hour 1
# ---------------------------------------
# Pick first timestep
t0 = network_nodes.snapshots[0]  # first time step
print(f"The imbalances for the first hour {t0}: ")
# Sum generation per bus
gen_per_bus = network_nodes.generators_t.p.loc[t0].groupby(network_nodes.generators.bus).sum()
# Sum load per bus
load_per_bus = network_nodes.loads_t.p.loc[t0].groupby(network_nodes.loads.bus).sum()
# Imbalance = generation - load at each bus
imbalances = gen_per_bus - load_per_bus

for bus, injection in imbalances.items():
    print(f"{bus}: {injection:.2f} MWh")



# Power flows in first hour
# ---------------------------------------

power_flow = network_nodes.lines_t.p0.loc[t0]

print("The powerflows for the first hour of the year are:")
print(power_flow)

# Plot that shows import and export in time step 1 in a plot
net_export_t0 = {}
for line in network_nodes.lines.index:
    bus0 = network_nodes.lines.loc[line, 'bus0']
    bus1 = network_nodes.lines.loc[line, 'bus1']
    flow = network_nodes.lines_t.p0.loc[t0, line]
    net_export_t0[bus0] = net_export_t0.get(bus0, 0) + flow
    net_export_t0[bus1] = net_export_t0.get(bus1, 0) - flow
plt.bar(net_export_t0.keys(), net_export_t0.values(), color='skyblue')
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel('Net export (MW, positive = exporter)')
plt.title('Net Export per Country (First Hour)')
plt.show()

# Plot first-hour trade and flow figures (bars + map).
importlib.reload(pf)
pf.plot_first_hour_trade_and_flow(network_nodes, t0=t0, show_demand_generation=True, extent=(1, 15, 40, 55))



