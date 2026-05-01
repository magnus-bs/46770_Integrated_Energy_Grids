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

import seaborn as sns
sns.set_theme(style="whitegrid", font="serif",
              rc={
                  "figure.figsize": (12, 5),
                  "font.size": 16,           # base font size
                  "axes.titlesize": 16,       # plot title
                  "axes.labelsize": 16,       # x and y axis labels
                  "xtick.labelsize": 14,      # x tick labels
                  "ytick.labelsize": 14,      # y tick labels
                  "legend.fontsize": 12,      # legend text
                  'axes.edgecolor': 'black',
                  'axes.spines.left': True,
                  'axes.spines.bottom': True,
                  'axes.spines.right': False,
                  'axes.spines.top': False,
                  'axes.grid': True,
                  'grid.color': "black",
                  'grid.linestyle': ':',
                  'grid.alpha': 0.3
              })








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
tech_colors = {
    'demand': 'black',
    "onshorewind": "#4C78A8",  # muted blue
    "solar": "#F2A541",        # soft orange
    "OCGT": "#E45756",          # soft red
    "nuclear": "#9D755D",      # brown
    "hydro": "#72B7B2"         # teal
}

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


# OCGT emissins
OCGT_efficiency = 0.39
OCGT_emission_th = 0.19 # t_CO2/MWh_th
OCGT_emission_el = OCGT_emission_th / OCGT_efficiency

# Add carriers
network.add("Carrier", "gas", co2_emissions=OCGT_emission_el, overwrite=True)  # in t_CO2/MWh_th
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
fuel_cost = 21.6  # in €/MWh_th #
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

#%%
print("\n=== Installed Capacities (GW) ===")
print(round(network.generators.p_nom_opt*10**(-3),2))

print("\n=== Total Dispatch (GWh over the year) ===")
print(round(network.generators_t.p.sum()*10**(-3),0))

# ----------------------------------------------------------------------------------------------
#                                 STEP 1: Visualisation of Results
### ----------------------------------------------------------------------------------------------
import importlib
import plot_functions as pf

print("Results of Step 1: 1-Node France Network")

# Weekly Disptach (summer/winter)
print("Winter:")
pf.weekly_dispatch_plot(network, tech_colors, start_day = 0, figsize = (13,3))
print("Summer:")
pf.weekly_dispatch_plot(network, tech_colors, start_day = 24*30*6, figsize=(13,3))

colors = [tech_colors['onshorewind'], tech_colors['solar'], tech_colors['OCGT'], tech_colors['nuclear']]
labels = ['Onshore Wind', 'Solar', 'Gas (OCGT)', 'Nuclear']

#print("Electricity Mix")
pf.capacity_dispatch_bars(network, tech_colors)
#pf.energy_mix_piechart(network, colors, labels, full_year = True, dpi = 300)
    
print("Load Duration Curves")
pf.duration_curves(network, tech_colors, figsize = (7,5.5), dpi = 400)




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
CF_wind_dict = {}
CF_solar_dict = {}

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

    # Save average capacity factor
    CF_wind_dict[year] = CF_wind
    CF_solar_dict[year] = CF_solar
    

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
#%% VISUALISATIONS:
# --------------
print("Analysis of Inter-Annual Variability")

#pf.annual_capacities_and_dispatch(df_results, tech_colors)

largest_difference = df_costs['total_cost_M€'].max() / df_costs['total_cost_M€'].min() * 100 - 100
print("The largest percentage difference between worst and best year is "+str(round(largest_difference,2))+"%")

pf.interannual_var_boxplots(df_results)

fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)

# Primary axis - costs
ax1.plot(df_costs['year'], df_costs['total_cost_M€'], marker='o', linestyle='-', color='tab:blue', label='Total Cost')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total System Cost (M€)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(alpha=0.3)

plt.title('Total System Cost per Weather Year')
plt.tight_layout()
plt.show()

print("Wind CF")
for year in years:
    print(year,":", CF_wind_dict[year].mean())
print("Solar CF")
for year in years:
    print(year,":", CF_solar_dict[year].mean())

importlib.reload(pf)
pf.corr_cf_load(CF_wind_dict, CF_solar_dict, demand)
    


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
#%%
start_day = 24 * 30 * 6 + 24 * 10
pf.weekly_dispatch_plot(network_storage, tech_colors, start_day, storage = True, figsize=(13,3), dpi = 600, ncol = 6)

pf.energy_mix_piechart(network_storage, colors, labels, full_year = True)

# 4. Plot State of Charge (How full the "battery" is)
pf.weekly_soc_plot(network_storage, start_day, full_year=True)



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

importlib.reload(pf)

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

# Prices (dual of nodal balance)    x
print(network_nodes.buses_t.marginal_price)


# Total system cost
print(f"Total system cost: {network_nodes.objective / 1e6:.2f} million €")


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


#%%
importlib.reload(pf)
pf.gen_cap_mix_stacked(df_gen_bus, df_cap_bus, df_demand, carrier_colors)




#%%
# ------------------------------------- II --------------------------------------
# Exports/Imports per country

print("Export Illustrations")
importlib.reload(pf)

pf.avg_annual_net_export_bar_plot(network_nodes)
pf.flow_matrix_heatmap(network_nodes)

pf.export_plot_analyses(network_nodes, week = slice('2015-07-10', '2015-07-17'))



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






#%% ----------------------------------------------------------------------------------------------
#                                       Step F: Impose CO2 Limit
### ----------------------------------------------------------------------------------------------


gen = network.generators_t.p
emissions_factor = network.generators.carrier.map(
    network.carriers.co2_emissions
).fillna(0)

# total emissions per generator (ton CO2)
gen_emissions = gen.multiply(emissions_factor, axis=1)

total_co2 = gen_emissions.sum().sum()

print(f"\n=== BASELINE CO2 EMISSIONS ===")
print(f"Total system emissions: {total_co2/1e6:.2f} Mton CO2")



co2_limits = [1e10, 3e7, 2.5e7, 2e7, 1.5e7, 1e7, 5e6, 1e6]

energy_mix = []
capacity_mix = []  # NEW: store capacities

for limit in co2_limits:
    network.model.solver_model = None

    n = network.copy()
    
    n.add("GlobalConstraint", "co2_limit",
          type="primary_energy",
          carrier_attribute="co2_emissions",
          sense="<=",
          constant=limit)

    n.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

    # -------------------
    # ENERGY (unchanged)
    # -------------------
    gen_energy = n.generators_t.p.sum()
    gen_by_carrier = gen_energy.groupby(n.generators.carrier).sum()
    energy_mix.append(gen_by_carrier)

    # -------------------
    # CAPACITY (NEW)
    # -------------------
    gen_capacity = n.generators.p_nom_opt
    cap_by_carrier = gen_capacity.groupby(n.generators.carrier).sum()
    capacity_mix.append(cap_by_carrier)




#%%
# =========================
# ENERGY MIX (PERCENTAGE)
# =========================
df_energy = pd.DataFrame(energy_mix, index=co2_limits)
df_energy = df_energy.fillna(0)
df_energy = df_energy / df_energy.sum(axis=1).values[:, None]
df_energy_percent = df_energy * 100
df_energy_percent.index = df_energy_percent.index / 1e6  # convert to Mton
idx = (df_energy_percent.index / 1e6).tolist()
idx = df_energy_percent.index.tolist()
idx[0] = r"$\infty$"
df_energy_percent.index = idx
carrier_color_map = {
    "onshorewind": tech_colors["onshorewind"],
    "solar": tech_colors["solar"],
    "gas": tech_colors["OCGT"],
    "nuclear": tech_colors["nuclear"]
}

df_energy_percent = df_energy_percent[carrier_color_map.keys()]

df_energy_percent.rename(columns={
    "onshorewind": "Onshore Wind",
    "solar": "Solar",
    "gas": "Gas (OCGT)",
    "nuclear": "Nuclear"
}, inplace=True)

# =========================
# CAPACITY MIX (ABSOLUTE)
# =========================
df_capacity = pd.DataFrame(capacity_mix, index=co2_limits)
df_capacity = df_capacity.fillna(0)

df_capacity = df_capacity[carrier_color_map.keys()]
df_capacity.index = df_capacity.index / 1e6  # convert to Mton
df_capacity = df_capacity / 1000  # convert to GW
idx = df_capacity.index.tolist()
idx[0] = r"$\infty$"
df_capacity.index = idx
df_capacity.rename(columns={
    "onshorewind": "Onshore Wind",
    "solar": "Solar",
    "gas": "Gas (OCGT)",
    "nuclear": "Nuclear"
}, inplace=True)


# =========================
# STYLE SETTINGS (ADJUST HERE)
# ===========================
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 14
X_TICK_SIZE = 16
LEGEND_SIZE = 16

# =========================
# PLOTTING
# =========================
fig, axes = plt.subplots(2, 1, figsize=(9,8), sharex=True, dpi = 300)

# -------------------------
# ENERGY MIX (TOP)
# -------------------------
df_energy_percent.plot(
    kind="bar",
    stacked=True,
    ax=axes[0],
    color=[carrier_color_map[k] for k in carrier_color_map.keys()],
    legend=False,
    alpha = 0.9
)

axes[0].set_ylabel("Energy Share (%)", fontsize=LABEL_SIZE)
axes[0].set_title("Energy Mix Evolution", fontsize=TITLE_SIZE)
axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
axes[0].grid(axis='y', alpha=0.3)


# -------------------------
# CAPACITY MIX (BOTTOM)
# -------------------------
df_capacity.plot(
    kind="bar",
    stacked=True,
    ax=axes[1],
    color=[carrier_color_map[k] for k in carrier_color_map.keys()],
    alpha = 0.9
)

axes[1].set_xlabel("CO2 Limit (Mton)", fontsize=LABEL_SIZE)
axes[1].set_ylabel("Installed Capacity (GW)", fontsize=LABEL_SIZE)
axes[1].set_title("Installed Capacity Evolution", fontsize=TITLE_SIZE)
axes[1].tick_params(axis='y',labelsize=TICK_SIZE)
axes[1].tick_params(axis='x', rotation=0,labelsize=X_TICK_SIZE)
axes[1].grid(axis='y', alpha=0.3)


# -------------------------
# SHARED LEGEND BELOW
# -------------------------
handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Technology",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_SIZE
)

axes[1].legend().remove()  # remove duplicate legend
ax = axes[1]

labels = ax.get_xticklabels()

for lab in labels:
    if lab.get_text() == r"$\infty$":
        lab.set_fontsize(28)
        lab.set_fontweight("bold")

# -------------------------
# FINAL LAYOUT
# -------------------------
plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave space for legend
plt.show()


# ----------------------------------------------------------------------------------------------
# Example results:
cap_15 = df_capacity.loc[15].sum()
cap_infty = df_capacity.loc["$\infty$"].sum()
pct_increase = (cap_15 / cap_infty - 1) * 100 
print("Increase in installed capacity with 15Mton CO2 limit: ",round(pct_increase,1),"%")

cap_1 = df_capacity.loc[1].sum()
pct_increase = (cap_1 / cap_infty - 1) * 100 
print("Increase in installed capacity with 1Mton CO2 limit: ",round(pct_increase,1),"%")




#%% ----------------------------------------------------------------------------------------------
#                                       Step G: Model Gas Pipes
### ----------------------------------------------------------------------------------------------


network_nodes.model.solver_model = None

n_gas_nodes = network_nodes.copy()

# Add gas storage in all countries
STORAGE = False

# Remove all old OCGT generators
ocgt_gens = n_gas_nodes.generators.index[
    n_gas_nodes.generators.carrier == "gas"
]

n_gas_nodes.remove("Generator", ocgt_gens)

for country in ["FR", "DE", "CH", "IT", "BE"]:
    n_gas_nodes.add("Bus", f"{country}_gas", carrier="gas")

# Add gas supply from Germany only
n_gas_nodes.add("Generator", "DE_gas_supply",
    bus="DE_gas",
    carrier="gas",
    p_nom_extendable=True,
    marginal_cost=fuel_cost  # €/MWh_th
)

capital_cost_OCGT_link = capital_cost_OCGT * efficiency  # scale to €/MW_th input

# Add OCGT as links between gas buses and electricity buses
n_gas_nodes.add("Link", "FR_OCGT", bus0="FR_gas", bus1="FR", efficiency=0.39, p_nom_extendable=True, capital_cost=capital_cost_OCGT_link, marginal_cost=0)
n_gas_nodes.add("Link", "BE_OCGT", bus0="BE_gas", bus1="BE", efficiency=0.39, p_nom_extendable=True, capital_cost=capital_cost_OCGT_link, marginal_cost=0)
n_gas_nodes.add("Link", "IT_OCGT", bus0="IT_gas", bus1="IT", efficiency=0.39, p_nom_extendable=True, capital_cost=capital_cost_OCGT_link, marginal_cost=0)
n_gas_nodes.add("Link", "CH_OCGT", bus0="CH_gas", bus1="CH", efficiency=0.39, p_nom_extendable=True, capital_cost=capital_cost_OCGT_link, marginal_cost=0)
n_gas_nodes.add("Link", "DE_OCGT",bus0="DE_gas", bus1="DE", efficiency=0.39, p_nom_extendable=True, capital_cost=capital_cost_OCGT_link, marginal_cost=0)

# capacities in GWh/day → convert to MW_th
gas_multiplier = 10000.3

gas_links = {
    ("DE_gas", "FR_gas"): 455*gas_multiplier,
    ("FR_gas", "DE_gas"): 180*gas_multiplier,
    ("CH_gas", "FR_gas"): 100*gas_multiplier,
    ("FR_gas", "CH_gas"): 260*gas_multiplier,
    ("FR_gas", "BE_gas"): 271*gas_multiplier,
    ("BE_gas", "FR_gas"): 462*gas_multiplier,
    ("IT_gas", "CH_gas"): 440*gas_multiplier,
    ("CH_gas", "IT_gas"): 640*gas_multiplier,
    ("CH_gas", "DE_gas"): 327*gas_multiplier,
    ("DE_gas", "CH_gas"): 411*gas_multiplier,
    ("DE_gas", "BE_gas"): 441*gas_multiplier,
    ("BE_gas", "DE_gas"): 435*gas_multiplier,
    ("DE_gas", "IT_gas"): 1192*gas_multiplier,
    ("IT_gas", "DE_gas"): 264* gas_multiplier
}

for i, ((b0, b1), cap_gwh_day) in enumerate(gas_links.items()):
    cap_mw = cap_gwh_day * 1e3 / 24  # convert to MW_th

    n_gas_nodes.add(
        "Link",
        f"{b0}_{b1}_gas",
        bus0=b0,
        bus1=b1,
        p_nom=cap_mw,
        efficiency=1.0,
        marginal_cost=0.000 # Minimal cost to ensure no circular flows (DE->CH->FR->DE for example)
    )

if STORAGE:
        # Gas storage parameters (example values - adjust!)
    gas_storage_capacity = {
        "DE": 0,  # They are modelled to be supplying country
        "FR": 125 * 10**6,
        "CH": 0, # Switzerland has very limited gas storage
        "IT": 200 * 10**6,
        "BE": 9 * 10**6
    }

    for country in ["FR", "DE", "CH", "IT", "BE"]:
        n_gas_nodes.add(
            "Store",
            f"{country}_gas_storage",
            bus=f"{country}_gas",
            e_nom=gas_storage_capacity[country],
            e_initial=0.5 * gas_storage_capacity[country], # realistic initial level
            e_cyclic=True,
            e_min_pu=0.1, # minimum 10% full to reflect operational constraints
            marginal_cost=0.0
        )

n_gas_nodes.carriers.at["gas", "co2_emissions"] = OCGT_emission_th

n_gas_nodes.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})


electricity_flow = n_gas_nodes.lines_t.p0.abs().sum().sum()

gas_links_names = [f"{b0}_{b1}_gas" for (b0, b1) in gas_links.keys()]
gas_flow = n_gas_nodes.links_t.p0[gas_links_names].abs().sum().sum()


print("Electricity flow (TWh):", round(electricity_flow*1e-6,0))
print("Gas flow (TWh):", round(gas_flow*1e-6,0))

# Gas flow on each pipeline
print("\nGas flow on pipelines (MWh_th):")
for i in range(len(gas_links)):
    flow = n_gas_nodes.links_t.p0[f"{list(gas_links.keys())[i][0]}_{list(gas_links.keys())[i][1]}_gas"].sum()
    print(f"Pipeline {list(gas_links.keys())[i][0]} <-> {list(gas_links.keys())[i][1]}: {flow*1e-6:.2f} TWh_th")

# Electricity flow on each interconnection
print("\nElectricity flow on interconnections (TWh):")
for line in n_gas_nodes.lines.index:
    flow = n_gas_nodes.lines_t.p0[line].sum()
    print(f"Line {line}: {flow*1e-6:.2f} TWh")

print("\n=== COSTS WITH GAS PIPELINES ===")
print(f"Total system cost: {n_gas_nodes.objective / 1e6:.2f} million €")





#%% ----------------------------------------------------------------------------------------------
#                                       Visualisations of Gas Network


# Stacked bar: total annual generation per technology per country

gen_by_bus = {}
gen_by_bus = {}

# -------------------------
# 1. Normal generators
# -------------------------
for gen in n_gas_nodes.generators.index:
    bus = n_gas_nodes.generators.loc[gen, 'bus']
    carrier = n_gas_nodes.generators.loc[gen, 'carrier']
    
    # Skip gas supply
    if carrier == "gas":
        continue
    
    total = float(n_gas_nodes.generators_t.p[gen].sum())
    
    gen_by_bus.setdefault(bus, {})
    gen_by_bus[bus][carrier] = gen_by_bus[bus].get(carrier, 0) + total

# -------------------------
# 2. OCGT (from links)
# -------------------------
for link in n_gas_nodes.links.index:
    if "OCGT" in link:
        bus = n_gas_nodes.links.loc[link, 'bus1']  # electricity side
        
        # p1 = electricity output
        total = -float(n_gas_nodes.links_t.p1[link].sum())
        
        gen_by_bus.setdefault(bus, {})
        gen_by_bus[bus]["gas"] = gen_by_bus[bus].get("gas", 0) + total

df_gen_bus = pd.DataFrame(gen_by_bus).T.fillna(0) / 1e6  # TWh

# Installed capacity per country and technology
cap_by_bus = {}
for gen in n_gas_nodes.generators.index:
    bus = n_gas_nodes.generators.loc[gen, 'bus']
    carrier = n_gas_nodes.generators.loc[gen, 'carrier']

    if carrier == "gas":
        continue

    cap = float(n_gas_nodes.generators.p_nom_opt[gen])
    cap_by_bus.setdefault(bus, {})
    cap_by_bus[bus][carrier] = cap_by_bus[bus].get(carrier, 0) + cap

# Add OCGT capacities from links
for link in n_gas_nodes.links.index:
    if "OCGT" in link:
        bus = n_gas_nodes.links.loc[link, 'bus1']
        cap = float(n_gas_nodes.links.p_nom_opt[link]) * n_gas_nodes.links.loc[link, 'efficiency']        
        cap_by_bus.setdefault(bus, {})
        cap_by_bus[bus]["gas"] = cap_by_bus[bus].get("gas", 0) + cap

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




#%% Emissions France

gen = network.generators_t.p
emissions_factor = network.generators.carrier.map(
    network.carriers.co2_emissions
).fillna(0)

# total emissions per generator (ton CO2)
gen_emissions = gen.multiply(emissions_factor, axis=1)

total_co2 = gen_emissions.sum().sum()

print(f"\n=== FRANCE CO2 EMISSIONS ===")
print(f"Total system emissions: {total_co2/1e6:.2f} Mton CO2")








#%% ----------------------------------------------------------------------------------------------
#                                       Step H: Impose CO2 Limit for Nodal System
### ----------------------------------------------------------------------------------------------


gen = network_nodes.generators_t.p
emissions_factor = network_nodes.generators.carrier.map(
    network_nodes.carriers.co2_emissions
).fillna(0)

# total emissions per generator (ton CO2)
gen_emissions = gen.multiply(emissions_factor, axis=1)

total_co2 = gen_emissions.sum().sum()

print(f"\n=== BASELINE CO2 EMISSIONS ===")
print(f"Total system emissions: {total_co2/1e6:.2f} Mton CO2")

co2_limit = 2.3e8
energy_mix = []
capacity_mix = []  # NEW: store capacities

network_nodes.model.solver_model = None

n_nodes_co2 = network_nodes.copy()

n_nodes_co2.add("GlobalConstraint", "co2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limit)

n_nodes_co2.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

# -------------------
# ENERGY (unchanged)
# -------------------
gen_energy = n_nodes_co2.generators_t.p.sum()
gen_by_carrier = gen_energy.groupby(n_nodes_co2.generators.carrier).sum()
energy_mix.append(gen_by_carrier)

# -------------------
# CAPACITY (NEW)
# -------------------
gen_capacity = n_nodes_co2.generators.p_nom_opt
cap_by_carrier = gen_capacity.groupby(n_nodes_co2.generators.carrier).sum()
capacity_mix.append(cap_by_carrier)


# Power flows
print(n_nodes_co2.lines_t.p0)  # power flow from bus0 to bus1

# Generator dispatch
print(n_nodes_co2.generators_t.p)

# Prices (dual of nodal balance)
print(n_nodes_co2.buses_t.marginal_price)


# Stacked bar: total annual generation per technology per country

gen_by_bus = {}
for gen in n_nodes_co2.generators.index:
    bus = n_nodes_co2.generators.loc[gen, 'bus']
    carrier = n_nodes_co2.generators.loc[gen, 'carrier']
    total = float(n_nodes_co2.generators_t.p[gen].sum())
    gen_by_bus.setdefault(bus, {})
    gen_by_bus[bus][carrier] = gen_by_bus[bus].get(carrier, 0) + total

df_gen_bus = pd.DataFrame(gen_by_bus).T.fillna(0) / 1e6  # TWh

# Installed capacity per country and technology
cap_by_bus = {}
for gen in n_nodes_co2.generators.index:
    bus = n_nodes_co2.generators.loc[gen, 'bus']
    carrier = n_nodes_co2.generators.loc[gen, 'carrier']
    cap = float(n_nodes_co2.generators.p_nom_opt[gen])
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

co2_shadow_price = n_nodes_co2.global_constraints.mu["co2_limit"]

print(f"CO2 shadow price: {co2_shadow_price:.2f} €/ton CO2")


# %%
