#%%
import pandas as pd
import pypsa
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#%% #####################################
### STEP 1 
#########################################

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define consistent colors for technologies
tech_colors = {
    'demand': 'black',
    'onshorewind': 'blue',
    'solar': 'orange',
    'OCGT': 'red',
    'nuclear': 'brown'
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
network.add("Bus", "electricity bus", y=46.2, x=2.2, v_nom=400, carrier="AC")


# Load electricity demand data
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)  # in MWh
df_elec.index = pd.to_datetime(df_elec.index)
country = 'FRA'

# Add load to the bus
network.add("Load", "load", bus="electricity bus", p_set=df_elec[country].values)

# Add carriers
network.add("Carrier", "gas", co2_emissions=0.19, overwrite=True)  # in t_CO2/MWh_th
network.add("Carrier", "nuclear", overwrite=True)
network.add("Carrier", "onshorewind", overwrite=True)
network.add("Carrier", "solar", overwrite=True)

# Add onshore wind generator
df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
CF_wind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_onshorewind = annuity(30, 0.07) * 910000 * (1 + 0.033)  # in €/MW
network.add("Generator", "onshorewind", bus="electricity bus", p_nom_extendable=True,
            carrier="onshorewind", capital_cost=capital_cost_onshorewind, marginal_cost=0,
            p_max_pu=CF_wind.values, overwrite=True)

# Add solar PV generator
df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)
CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_solar = annuity(25, 0.07) * 425000 * (1 + 0.03)  # in €/MW
network.add("Generator", "solar", bus="electricity bus", p_nom_extendable=True,
            carrier="solar", capital_cost=capital_cost_solar, marginal_cost=0,
            p_max_pu=CF_solar.values, overwrite=True)

# Add OCGT generator
capital_cost_OCGT = annuity(25, 0.07) * 560000 * (1 + 0.033)  # in €/MW
fuel_cost = 21.6  # in €/MWh_th
efficiency = 0.39
marginal_cost_OCGT = fuel_cost / efficiency  # in €/MWh_el
network.add("Generator", "OCGT", bus="electricity bus", p_nom_extendable=True,
            carrier="gas", capital_cost=capital_cost_OCGT, marginal_cost=marginal_cost_OCGT,
            overwrite=True)


# Add nuclear generator
average_demand = df_elec[country].mean()  # in MW
capital_cost_nuclear = annuity(60, 0.07) * 3460 * 1000 * (1 + 0.03)
network.add("Generator", "nuclear", bus="electricity bus", p_nom_extendable=True,
            carrier="nuclear", capital_cost=capital_cost_nuclear, marginal_cost=6,
            p_nom_max=average_demand*0.7, p_max_pu=1.0, p_min_pu=0.5, ramp_limit_up=0.15, ramp_limit_down=0.15,
            overwrite=True)

# Optimize without CO2 limit
network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

print("Optimization Results (No CO2 Limit):")
print(f"Total cost: {network.objective / 1000000:.2f} million €")
print(f"Levelized cost: {network.objective / float(network.loads_t.p.sum()):.2f} €/MWh")
print("Optimal capacities (MW):")
for gen in network.generators.index:
    print(f"{gen}: {float(network.generators.p_nom_opt[gen]):.2f}")


# Plot dispatch (In winter, first week)
plot_time=24*7
plt.figure(figsize=(10, 6))
plt.plot(network.loads_t.p['load'][0:plot_time], color=tech_colors['demand'], label='demand')
plt.plot(network.generators_t.p['onshorewind'][0:plot_time], color=tech_colors['onshorewind'], label='onshore wind')
plt.plot(network.generators_t.p['solar'][0:plot_time], color=tech_colors['solar'], label='solar')
plt.plot(network.generators_t.p['OCGT'][0:plot_time], color=tech_colors['OCGT'], label='gas (OCGT)')
plt.plot(network.generators_t.p['nuclear'][0:plot_time], color=tech_colors['nuclear'], label='Nuclear')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.title('Electricity Dispatch (First 196 Hours in winter)')
plt.xlabel('Hour')
plt.ylabel('Power (MW)')
plt.show()

# Plot dispatch (In summer)
plot_summer_delay=24*30*6
plt.figure(figsize=(10, 6))
plt.plot(network.loads_t.p['load'][plot_summer_delay:plot_summer_delay+plot_time], color=tech_colors['demand'], label='demand')
plt.plot(network.generators_t.p['onshorewind'][plot_summer_delay:plot_summer_delay+plot_time], color=tech_colors['onshorewind'], label='onshore wind')
plt.plot(network.generators_t.p['solar'][plot_summer_delay:plot_summer_delay+plot_time], color=tech_colors['solar'], label='solar')
plt.plot(network.generators_t.p['OCGT'][plot_summer_delay:plot_summer_delay+plot_time], color=tech_colors['OCGT'], label='gas (OCGT)')
plt.plot(network.generators_t.p['nuclear'][plot_summer_delay:plot_summer_delay+plot_time], color=tech_colors['nuclear'], label='Nuclear')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.title('Electricity Dispatch (In summer)')
plt.xlabel('Hour')
plt.ylabel('Power (MW)')
plt.show()

# Pie chart for energy mix
labels = ['onshore wind', 'solar', 'gas (OCGT)', 'Nuclear']
sizes = [float(network.generators_t.p['onshorewind'].sum()),
         float(network.generators_t.p['solar'].sum()),
         float(network.generators_t.p['OCGT'].sum()),
         float(network.generators_t.p['nuclear'].sum())]
colors = [tech_colors['onshorewind'], tech_colors['solar'], tech_colors['OCGT'], tech_colors['nuclear']]
plt.figure()
plt.pie(sizes, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Electricity Mix (Full year)', y=1.07)
plt.show()

"""
# Define winter and summer months
winter_months = [1, 2, 12]  # Jan, Feb, Dec
summer_months = [6, 7, 8]   # Jun, Jul, Aug

snapshots = pd.DatetimeIndex(network.snapshots)

winter_snaps = snapshots[snapshots.month.isin(winter_months)]
summer_snaps = snapshots[snapshots.month.isin(summer_months)]

# Sum generation for winter and summer
winter_gen = network.generators_t.p.loc[winter_snaps].sum()
summer_gen = network.generators_t.p.loc[summer_snaps].sum()

# Prepare sizes for pie charts
winter_sizes = [
    float(winter_gen['onshorewind']),
    float(winter_gen['solar']),
    float(winter_gen['OCGT']),
    float(winter_gen['nuclear'])
]

summer_sizes = [
    float(summer_gen['onshorewind']),
    float(summer_gen['solar']),
    float(summer_gen['OCGT']),
    float(summer_gen['nuclear'])
]

# Pie chart labels and colors
labels = ['onshore wind', 'solar', 'gas (OCGT)', 'Nuclear']
colors = [tech_colors['onshorewind'], tech_colors['solar'], tech_colors['OCGT'], tech_colors['nuclear']]

# --- Winter electricity mix ---
plt.figure()
plt.pie(winter_sizes, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Electricity Mix (Winter)', y=1.07)
plt.show()

# --- Summer electricity mix ---
plt.figure()
plt.pie(summer_sizes, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Electricity Mix (Summer)', y=1.07)
plt.show()
"""

# Load duration curves
plt.figure(figsize=(10,6))

for gen in network.generators_t.p.columns:
    sorted_values = np.sort(network.generators_t.p[gen].values)[::-1]  # sort descending
    hours = np.arange(1, len(sorted_values)+1)
    plt.plot(hours, sorted_values, label=gen, color=tech_colors.get(gen, 'gray'))

plt.xlabel("Hours per year (sorted)")
plt.ylabel("Power output (MW)")
plt.title("Annual Generation Duration Curve")
plt.legend(fancybox=True, shadow=True, loc='best')
plt.grid(True, alpha=0.3)
plt.show()


#%% #####################################
### STEP 2 
#########################################


demand = df_elec[country].values  # will be reused for all weather years

# Onshore wind
df_wind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_wind.index = pd.to_datetime(df_wind.index, utc=True)  # parse as UTC first
df_wind.index = df_wind.index.tz_convert(None)           # remove timezone

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index, utc=True)
df_solar.index = df_solar.index.tz_convert(None)

years = range(2010, 2018)

results = []
cost_results = []


for year in years:
    print(f"Running optimization for {year}...")
    
    # Filter weather data for the year
    # Filter weather data for the year
    CF_wind = df_wind.loc[f'{year}-01-01':f'{year}-12-31', country]
    CF_solar = df_solar.loc[f'{year}-01-01':f'{year}-12-31', country]

    # Drop Feb 29 if present (leap year)
    CF_wind = CF_wind[~((CF_wind.index.month == 2) & (CF_wind.index.day == 29))]
    CF_solar = CF_solar[~((CF_solar.index.month == 2) & (CF_solar.index.day == 29))]

    # Convert to values
    CF_wind = CF_wind.values
    CF_solar = CF_solar.values

    # Create network
    network_year = pypsa.Network()
    network_year.set_snapshots(pd.RangeIndex(len(demand)))
    network_year.add("Bus", "electricity bus", p_set=demand)
    
    # Load (same for all years)
    network_year.add("Load", "load", bus="electricity bus", p_set=demand)

    # Add carriers
    network_year.add("Carrier", "gas", co2_emissions=0.19, overwrite=True)
    network_year.add("Carrier", "nuclear", overwrite=True)
    network_year.add("Carrier", "onshorewind", overwrite=True)
    network_year.add("Carrier", "solar", overwrite=True)
    
    # Add generators
    network_year.add("Generator", "onshorewind", bus="electricity bus", p_nom_extendable=True,
                carrier="onshorewind", capital_cost=capital_cost_onshorewind, marginal_cost=0,
                p_max_pu=CF_wind, overwrite=True)

    network_year.add("Generator", "solar", bus="electricity bus", p_nom_extendable=True,
                carrier="solar", capital_cost=capital_cost_solar, marginal_cost=0,
                p_max_pu=CF_solar, overwrite=True)

    network_year.add("Generator", "OCGT", bus="electricity bus", p_nom_extendable=True,
                carrier="gas", capital_cost=capital_cost_OCGT, marginal_cost=marginal_cost_OCGT,
                overwrite=True)

    avg_demand = df_elec[country].mean()
    network_year.add("Generator", "nuclear", bus="electricity bus", p_nom_extendable=True,
                carrier="nuclear", capital_cost=capital_cost_nuclear, marginal_cost=6,
                p_nom_max=avg_demand*0.7, p_max_pu=1.0, p_min_pu=0.5,
                ramp_limit_up=0.15, ramp_limit_down=0.15, overwrite=True)
    # Optimize
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


# Average optimal capacity per year
plt.figure(figsize=(10,6))
sns.barplot(data=df_results, x='year', y='p_nom', hue='generator', palette=tech_colors)
plt.ylabel("Optimal Capacity (MW)")
plt.title("Optimal Generator Capacities for Different Weather Years")
plt.show()

# Variability of dispatch per year
plt.figure(figsize=(10,6))
sns.barplot(data=df_results, x='year', y='dispatch_std', hue='generator', palette=tech_colors)
plt.ylabel("Dispatch Standard Deviation (MW)")
plt.title("Generator Dispatch Variability for Different Weather Years")
plt.show()


# Stacked Bar Chart: Annual Dispatch per Technology per Year

# Sum total dispatch per year and per generator
dispatch_per_year = df_results.groupby(['year', 'generator'])['total_dispatch'].sum().unstack().fillna(0)

dispatch_per_year.plot(
    kind='bar',
    stacked=True,
    color=[tech_colors.get(col, 'gray') for col in dispatch_per_year.columns],
    figsize=(10,6)
)

plt.ylabel("Total Annual Dispatch (MWh)")
plt.xlabel("Year")
plt.title("Annual Electricity Dispatch per Technology")
plt.legend(title='Generator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Plot total over years
plt.figure(figsize=(10,6))
plt.plot(df_costs['year'], df_costs['total_cost_M€'], marker='o', linestyle='-', color='tab:blue')
plt.xlabel('Year')
plt.ylabel('Total System Cost (M€)')
plt.title('Total System Cost per Weather Year')
plt.grid(alpha=0.3)
#plt.ylim(0,df_costs['total_cost_M€'].max())
plt.show()

largest_difference = df_costs['total_cost_M€'].max() / df_costs['total_cost_M€'].min() * 100 - 100

print("The largest percentage difference between worst and best year is "+str(round(largest_difference,2))+"%")


#%% #######################################
###### STEP 3 #############################
###########################################

network.model.solver_model = None

network_storage = network.copy()

# Add storage unit
lifetime = 60
capital_cost_hydro = annuity(60, 0.07) * 1994 * 10**3
fixed_o_m = 16.46 * 10**3  # EUR/MW/yr
network_storage.add("StorageUnit", "Pumped Hydro", bus="electricity bus", p_nom_extendable=True,
            max_hours=11, efficiency_store=0.95, efficiency_dispatch=0.85,
            capital_cost=capital_cost_hydro + fixed_o_m, marginal_cost=0, overwrite=True)

print("\nStorage unit added: Pumped Hydro")

network_storage.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

print("\nOptimization Results (With Pumped Hydro):")
print(f"Total cost: {network_storage.objective / 1e6:.2f} million €")
print(f"Levelized cost: {network_storage.objective / float(network.loads_t.p.sum()):.2f} €/MWh")

# 2. Check optimal storage capacity
p_nom_storage = network_storage.storage_units.p_nom_opt["Pumped Hydro"]
print(f"Optimal Pumped Hydro Capacity: {p_nom_storage:.2f} MW")

# 3. Plot Dispatch with Storage (First week of the last year)
plot_time = 24 * 7
plt.figure(figsize=(12, 6))

# Plot Demand and Generation
plt.plot(network_storage.loads_t.p['load'][:plot_time], color='black', label='Demand', lw=2)
plt.plot(network_storage.generators_t.p['onshorewind'][:plot_time], color='blue', label='Onshore Wind', alpha=0.7)
plt.plot(network_storage.generators_t.p['solar'][:plot_time], color='orange', label='Solar', alpha=0.7)
plt.plot(network_storage.generators_t.p['nuclear'][:plot_time], color='brown', label='Nuclear', alpha=0.7)

# Plot Storage Dispatch (Positive is discharging, Negative is charging)
plt.fill_between(network_storage.snapshots[:plot_time], 
                 network_storage.storage_units_t.p[:plot_time]["Pumped Hydro"], 
                 color='teal', label='Storage Dispatch', alpha=0.5)

plt.legend(loc='upper right')
plt.title('System Dispatch with Pumped Hydro Storage (One Week)')
plt.ylabel('Power (MW)')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Plot State of Charge (How full the "battery" is)
plt.figure(figsize=(12, 4))
plt.plot(network_storage.storage_units_t.state_of_charge[:plot_time]["Pumped Hydro"], color='teal')
plt.title('Pumped Hydro State of Charge (MWh)')
plt.ylabel('Energy Stored (MWh)')
plt.fill_between(network_storage.snapshots[:plot_time], 
                 network_storage.storage_units_t.state_of_charge[:plot_time]["Pumped Hydro"], 
                 color='teal', alpha=0.2)
plt.grid(True, alpha=0.3)
plt.show()

# Pie chart for energy mix with storage
sizes_storage = [float(network_storage.generators_t.p['onshorewind'].sum()),
             float(network_storage.generators_t.p['solar'].sum()),
             float(network_storage.generators_t.p['OCGT'].sum()),
             float(network_storage.generators_t.p['nuclear'].sum())]
plt.figure()
plt.pie(sizes_storage, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Electricity Mix (With Storage)', y=1.07)
plt.show()












#%% ############################
## CO2 LIMIT ###################
################################


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
print(f"Levelized cost: {network_co2.objective / float(network_co2.loads_t.p.sum()):.2f} €/MWh")
print("Optimal capacities (MW):")
for gen in network_co2.generators.index:
    print(f"{gen}: {float(network_co2.generators.p_nom_opt[gen]):.2f}")

# Plot dispatch with CO2 limit
plt.figure(figsize=(10, 6))
plt.plot(network_co2.loads_t.p['load'][0:plot_time], color='black', label='demand')
plt.plot(network_co2.generators_t.p['onshorewind'][0:plot_time], color='blue', label='onshore wind')
plt.plot(network_co2.generators_t.p['solar'][0:plot_time], color='orange', label='solar')
plt.plot(network_co2.generators_t.p['OCGT'][0:plot_time], color='red', label='gas (OCGT)')
plt.plot(network_co2.generators_t.p['nuclear'][0:plot_time], color='brown', label='Nuclear')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.title('Electricity Dispatch (First 96 Hours, With CO2 Limit)')
plt.xlabel('Hour')
plt.ylabel('Power (MW)')
plt.show()

# Pie chart for energy mix with CO2 limit
sizes_co2 = [
    float(network_co2.generators_t.p['onshorewind'].sum()),
    float(network_co2.generators_t.p['solar'].sum()),
    float(network_co2.generators_t.p['OCGT'].sum()),
    float(network_co2.generators_t.p['nuclear'].sum())
]

plt.figure()
plt.pie(sizes_co2, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Electricity Mix (With CO2 Limit)', y=1.07)
plt.show()



#%%#################################
# Step 4
####################################

# Add neighboring countries as buses
network.add("Bus", "DE", y=51.0, x=10.0, v_nom=400, carrier="AC")
network.add("Bus", "CH", y=46.8, x=8.3, v_nom=400, carrier="AC")
network.add("Bus", "IT", y=43.0, x=12.5, v_nom=400, carrier="AC")
network.add("Bus", "ES", y=40.4, x=-3.7, v_nom=400, carrier="AC")
network.add("Bus", "UK", y=51.5, x=-0.1, v_nom=400, carrier="AC")


# France connections
network.add("Line", "FR-CH", bus0="electricity bus", bus1="CH", s_nom=500, x=0.1, r=0)
network.add("Line", "FR-DE", bus0="electricity bus", bus1="DE", s_nom=500, x=0.1, r=0)
network.add("Line", "FR-ES", bus0="electricity bus", bus1="ES", s_nom=500, x=1, r=0)
network.add("Line", "FR-IT", bus0="electricity bus", bus1="IT", s_nom=500, x=1, r=0)
network.add("Line", "FR-UK", bus0="electricity bus", bus1="UK", s_nom=500, x=1, r=0)

# extra lines to create a cycles
network.add("Line", "CH-DE", bus0="CH", bus1="DE", s_nom=500, x=0.1, r=0)
network.add("Line", "CH-IT", bus0="CH", bus1="IT", s_nom=500, x=0.1, r=0)


# simple plot
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection=ccrs.PlateCarree())

network.plot(
    ax=ax,
    margin=0.2,
    bus_sizes=0.2,
    bus_colors="orange",
    bus_alpha=0.7,
    line_colors="orchid",
    line_widths=3,
    title="Connections",
)

ax.set_extent([-5, 15, 40, 55], crs=ccrs.PlateCarree())

plt.show()

# maybe more nice plot
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection=ccrs.PlateCarree())

# add map background
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
ax.add_feature(cfeature.COASTLINE)

network.plot(
    ax=ax,
    margin=0.2,
    bus_sizes=0.5,
    bus_colors="orange",
    bus_alpha=0.7,
    line_colors="orchid",
    line_widths=3,
    title="Connections",
)

ax.set_extent([-5, 15, 40, 55])
for bus in network.buses.index:
    x = network.buses.loc[bus, "x"]
    y = network.buses.loc[bus, "y"]
    ax.text(x, y, bus)

plt.show()

#%%
# Run optimization
network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

# Power flows
print(network.lines_t.p0)  # power flow from bus0 to bus1

# Generator dispatch
print(network.generators_t.p)

# Prices (dual of nodal balance)
print(network.buses_t.marginal_price)
# %%
# Line loading (how congested each line is)
line_loading = network.lines_t.p0.abs() / network.lines.s_nom * 100  # in %
line_loading.mean().plot(kind='bar', figsize=(10,5), title='Average Line Loading (%)')
plt.ylabel('Loading (%)')
plt.axhline(100, color='red', linestyle='--', label='Capacity limit')
plt.legend()
plt.show()

#%%
# Average marginal price per bus
avg_prices = network.buses_t.marginal_price.mean()
print(avg_prices)

# Price over time per bus
network.buses_t.marginal_price.plot(figsize=(12,5), title='Nodal Marginal Prices Over Time')
plt.ylabel('Price (€/MWh)')
plt.show()
# %%
