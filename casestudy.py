
#%%
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_functions as pf

#%%

#%% ----------------------------------------------------------------------------------------------
#                                 Create networks to compare
### ----------------------------------------------------------------------------------------------

# BASE NETWORK
#------------------------------
# Copy of nodal network created in main_file.py
network_base = pypsa.Network("nodal_network.nc")

# NEW NETWORK
#------------------------------
# Create a copy of the base network to modify
#network_base.model.solver_model = None
network_new=network_base.copy()

# Remove nuclear generator in france in new network
network_new.remove("Generator", "nuclear")

#%% ----------------------------------------------------------------------------------------------
#                                 Optimize both networks
### ----------------------------------------------------------------------------------------------

network_base.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})
network_new.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

#%% ----------------------------------------------------------------------------------------------
#                                 Compare results
### ----------------------------------------------------------------------------------------------

#%% -------------- Compare total system cost-------------------
print("\n=== Total System Cost (M€) ===")
print(f'Basecase: {round(network_base.objective/1e6,2)}')
# Add the annualised capital cost of the nuclear generator in france to the total system cost of the new network
# as we only close the  nuclear generator we have still paid the capital cost of the nuclear generator in france, 
# but we do not have any dispatch cost from the nuclear generator in france
total_cost_new = network_new.objective + network_base.generators.loc["nuclear", "capital_cost"] * network_base.generators.loc["nuclear", "p_nom_opt"]
print(f'New case: {round(total_cost_new/1e6,2)}')

#%% -------------- nodal prices -------------------

# Find nodal prices in both networks
prices_base = network_base.buses_t.marginal_price
prices_new = network_new.buses_t.marginal_price

# Find the 5 highest prices and their date
top5_base = prices_base["FR"].nlargest(5)
top5_new = prices_new["FR"].nlargest(5)
print("\n=== Top 5 highest prices in France (Base) ===")
for ts, price in top5_base.items():
    print(f"{ts}: {price:.2f} €/MWh")
print("\n=== Top 5 highest prices in France (New) ===")
for ts, price in top5_new.items():
    print(f"{ts}: {price:.2f} €/MWh")

# Find nodal prices in countries where we remove 2015-01-19 17:00:00 (the hour with the highest price in France) to avoid skewing the plot
prices_no_outlier_base = prices_base.copy()
prices_no_outlier_base = prices_no_outlier_base.drop([
    pd.Timestamp("2015-01-19 17:00:00"),
    pd.Timestamp("2015-01-20 17:00:00")
], errors="ignore")
prices_no_outlier_new = prices_new.copy()
prices_no_outlier_new = prices_no_outlier_new.drop([
    pd.Timestamp("2015-01-19 17:00:00"),
    pd.Timestamp("2015-01-20 17:00:00")
], errors="ignore")


# print average nodal price in each country in BASE network
print("\n=== BASE: Average Nodal Price per country (€/MWh) ===")
for bus in network_base.buses.index:
    avg_price = prices_base[bus].mean()
    print(f"{bus}: {avg_price:.2f} €/MWh")

# print average nodal price in each country in NEW network
print("\n===NEW: Average Nodal Price per country  (€/MWh) ===")
for bus in network_new.buses.index:
    avg_price = prices_new[bus].mean()
    print(f"{bus}: {avg_price:.2f} €/MWh")

# print average nodal price in each country in BASE network without outlier and standard variation
print("\n=== BASE: Average Nodal Price per country without outlier(€/MWh) ===")
for bus in network_base.buses.index:
    avg_price = prices_no_outlier_base[bus].mean()
    std_price = prices_no_outlier_base[bus].std()
    print(f"{bus}: {avg_price:.2f} ± {std_price:.2f} €/MWh")

# print average nodal price in each country in NEW network without outlier and standard variation
print("\n=== NEW: Average Nodal Price per country without outlier(€/MWh) ===")
for bus in network_new.buses.index:
    avg_price = prices_no_outlier_new[bus].mean()
    std_price = prices_no_outlier_new[bus].std()
    print(f"{bus}: {avg_price:.2f} ± {std_price:.2f} €/MWh")

#%% Find weighted average price across all countries (weighted by load) in both networks without outlier

load_by_bus_base = network_base.loads_t.p_set.groupby(network_base.loads.bus, axis=1).sum()
load_by_bus_new = network_new.loads_t.p_set.groupby(network_new.loads.bus, axis=1).sum()

weighted_price_base = float(
    prices_no_outlier_base.mul(load_by_bus_base).sum().sum() / load_by_bus_base.sum().sum()
)
weighted_price_new = float(
    prices_no_outlier_new.mul(load_by_bus_new).sum().sum() / load_by_bus_new.sum().sum()
)

# Print weighted average in each country in each system
print("\n=== Load-Weighted Average Nodal Price per country without outlier (€/MWh) ===")
for bus in network_base.buses.index:
    weighted_price_base = float(
        prices_no_outlier_base[bus].mul(load_by_bus_base[bus]).sum() / load_by_bus_base[bus].sum()
    )
    weighted_price_new = float(
        prices_no_outlier_new[bus].mul(load_by_bus_new[bus]).sum() / load_by_bus_new[bus].sum()
    )
    print(f"{bus}: Base: {weighted_price_base:.2f} €/MWh, New: {weighted_price_new:.2f} €/MWh") 


#%% Print how big percentage of the time each technology sets the price in each network (without outlier) in each coutry
def price_setting_shares(network, prices_no_outlier, tolerance=1e-3):
    shares_by_bus = {}

    for bus in network.buses.index:
        bus_generators = network.generators[network.generators.bus == bus]
        bus_prices = prices_no_outlier[bus]
        shares_by_technology = {}

        for carrier, carrier_generators in bus_generators.groupby("carrier"):
            marginal_cost = float(carrier_generators["marginal_cost"].iloc[0])
            share = float(np.isclose(bus_prices, marginal_cost, atol=tolerance).mean() * 100)
            shares_by_technology[carrier] = share

        shares_by_bus[bus] = shares_by_technology

    return shares_by_bus


print("\n=== Percentage of time each technology sets the price without outlier ===")
price_setting_base = price_setting_shares(network_base, prices_no_outlier_base)
price_setting_new = price_setting_shares(network_new, prices_no_outlier_new)

for bus in network_base.buses.index:
    print(f"\n{bus}")
    technologies = sorted(set(price_setting_base[bus]) | set(price_setting_new[bus]))

    for technology in technologies:
        base_share = price_setting_base[bus].get(technology, 0.0)
        new_share = price_setting_new[bus].get(technology, 0.0)
        print(f"{technology}: Base {base_share:.2f}%, New {new_share:.2f}%")


def get_cap_by_bus(network):
    cap_by_bus = {}

    for generator in network.generators.index:
        bus = network.generators.loc[generator, "bus"]
        carrier = network.generators.loc[generator, "carrier"]
        capacity = float(network.generators.p_nom_opt[generator])

        cap_by_bus.setdefault(bus, {})
        cap_by_bus[bus][carrier] = cap_by_bus[bus].get(carrier, 0.0) + capacity

    return pd.DataFrame(cap_by_bus).T.fillna(0) / 1000




# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
cap_base = get_cap_by_bus(network_base)
cap_new = get_cap_by_bus(network_new)

# align structure
cap_base, cap_new = cap_base.align(cap_new, fill_value=0)

# difference (KEY OUTPUT)
cap_diff = cap_new - cap_base

# --------------------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------------------
carrier_colors = {
    "onshorewind": "#4C78A8",
    "solar": "#F2A541",
    "gas": "#E45756",
    "nuclear": "#9D755D",
    "hydro": "#72B7B2"
}

# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --------------------------------------------------------------------------------------
# 1. BASE vs NEW stacked capacity
# --------------------------------------------------------------------------------------
bottom_base = pd.Series(0, index=cap_base.index)
bottom_new = pd.Series(0, index=cap_new.index)

for carrier in cap_base.columns:

    axes[0].bar(
        cap_base.index,
        cap_base[carrier],
        bottom=bottom_base,
        label=carrier,
        color=carrier_colors.get(carrier, None)
    )
    bottom_base += cap_base[carrier]

    axes[1].bar(
        cap_new.index,
        cap_new[carrier],
        bottom=bottom_new,
        label=carrier,
        color=carrier_colors.get(carrier, None)
    )
    bottom_new += cap_new[carrier]

axes[0].set_title("Base case installed capacity (GW)")
axes[1].set_title("New case installed capacity (GW)")

axes[0].set_ylabel("GW")

# --------------------------------------------------------------------------------------
# 2. Formatting
# --------------------------------------------------------------------------------------
for ax in axes:
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()



#%%

#------------------------------------------------------------------------#
# PRICE PLOT
#------------------------------------------------------------------------#

#%%import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import cartopy.io.shapereader as shpreader

# --------------------------------------------------------------------------------------
# Data (ISO3)
# --------------------------------------------------------------------------------------
iso3_map = {
    "FR": "FRA",
    "DE": "DEU",
    "CH": "CHE",
    "IT": "ITA",
    "BE": "BEL",
}

base = {
    iso3_map[bus]: float(prices_no_outlier_base[bus].mean())
    for bus in network_base.buses.index
}

new = {
    iso3_map[bus]: float(prices_no_outlier_new[bus].mean())
    for bus in network_new.buses.index
}

# shared color scale
all_values = list(base.values()) + list(new.values())
norm = mpl.colors.Normalize(vmin=min(all_values), vmax=max(all_values))
cmap = plt.get_cmap("Blues")

# --------------------------------------------------------------------------------------
# Load countries
# --------------------------------------------------------------------------------------
shp = shpreader.natural_earth(
    resolution='50m',
    category='cultural',
    name='admin_0_countries'
)
reader = shpreader.Reader(shp)

# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

cases = [base, new]
titles = ["Base case", "New case"]

for ax, data, title in zip(axes, cases, titles):

    # zoom to Europe
    ax.set_extent([-6, 20, 35, 60], crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    for country in reader.records():
        iso3 = country.attributes["ADM0_A3"]

        if iso3 in data:
            value = data[iso3]
            color = cmap(norm(value))

            # fill country
            ax.add_geometries(
                [country.geometry],
                ccrs.PlateCarree(),
                facecolor=color,
                edgecolor="black",
                linewidth=0.6
            )

            # ----------------------------------------------------------------------------------
            # Label INSIDE country (robust placement)
            # ----------------------------------------------------------------------------------
            point = country.geometry.representative_point()

            ax.text(
                point.x,
                point.y,
                f"{iso3}\n{value:.1f}",
                transform=ccrs.PlateCarree(),
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.2"
                )
            )

    ax.set_title(title)

# --------------------------------------------------------------------------------------
# FIXED COLORBAR (no overlap)
# --------------------------------------------------------------------------------------
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("€/MWh")

# layout spacing so colorbar doesn't overlap
plt.subplots_adjust(right=0.9)

plt.show()


#%%#########################################
tech_colors = {
    'demand': 'black',
    "onshorewind": "#4C78A8",  # muted blue
    "solar": "#F2A541",        # soft orange
    "OCGT": "#E45756",          # soft red
    "nuclear": "#9D755D",      # brown
    "hydro": "#72B7B2"         # teal
}
pf.capacity_dispatch_bars(network_base,tech_colors)
pf.capacity_dispatch_bars(network_new,tech_colors)

def capacity_dispatch_bar(network, tech_colors, country):

    # --- Select generators in the country (bus = country) ---
    gens = network.generators[network.generators.bus == country]

    # --- Capacity (GW) ---
    capacity = gens.p_nom_opt * 1e-3

    # --- Dispatch (GWh over full period) ---
    dispatch = network.generators_t.p[gens.index].sum() * 1e-3

    # --- Aggregate by technology ---
    cap_by_tech = capacity.groupby(gens.carrier).sum()
    disp_by_tech = dispatch.groupby(gens.carrier).sum()

    # --- Align technologies ---
    techs = cap_by_tech.index.union(disp_by_tech.index)
    cap_by_tech = cap_by_tech.reindex(techs).fillna(0)
    disp_by_tech = disp_by_tech.reindex(techs).fillna(0)

    # --- Convert to shares ---
    cap_pct = cap_by_tech / cap_by_tech.sum() * 100
    disp_pct = disp_by_tech / disp_by_tech.sum() * 100

    # --- Colors ---
    colors = [tech_colors.get(t, tech_colors['OCGT']) for t in techs]

    labels = {
        "onshorewind": "Onshore Wind",
        "solar": "Solar",
        "gas": "Gas (OCGT)",
        "nuclear": "Nuclear",
        "hydro": "Hydro"
    }

    # --- Plot setup ---
    y = np.arange(len(techs))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)

    # Capacity (upper bar)
    ax.barh(
        y + bar_height/2,
        cap_pct,
        height=bar_height,
        color=colors,
        hatch='///',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7,
        label="Capacity (%)"
    )

    # Dispatch (lower bar)
    ax.barh(
        y - bar_height/2,
        disp_pct,
        height=bar_height,
        color=colors,
        alpha=0.7,
        label="Dispatch (%)"
    )

    # --- Labels inside bars ---
    for i, (cap, disp) in enumerate(zip(cap_pct, disp_pct)):

        ax.text(cap / 2, i + bar_height/2,
                f"{cap:.1f}%", va='center', ha='center',
                fontsize=10, color='white', fontweight='bold')

        ax.text(disp / 2, i - bar_height/2,
                f"{disp:.1f}%", va='center', ha='center',
                fontsize=10, color='white', fontweight='bold')

    # --- Formatting ---
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    y_labels = [labels.get(t, t) for t in techs]
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Share (%)")
    #ax.set_title(f"Capacity vs Dispatch by Technology ({country})")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Capacity'),
        Patch(facecolor='white', edgecolor='black', label='Dispatch')
    ]

    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()


#%% 
    
capacity_dispatch_bar(network_base,tech_colors,country="BE")

capacity_dispatch_bar(network_new,tech_colors,country="BE")

#%%


def plot_stacked_case_comparison(plot_ax, frame, plot_title, y_label, carrier_colors_by_name):
    import matplotlib.patheffects as pe

    positions = np.arange(len(frame.index))
    bottoms = np.zeros(len(frame.index))

    for carrier_name in frame.columns:
        heights = frame[carrier_name].fillna(0).values
        plot_ax.bar(
            positions,
            heights,
            bottom=bottoms,
            color=carrier_colors_by_name.get(carrier_name, "#999999"),
            label=carrier_name,
            width=0.62,
            edgecolor="white",
            linewidth=0.7,
        )

        totals = frame.sum(axis=1).replace(0, np.nan).values
        for index, height in enumerate(heights):
            if height <= 0:
                continue
            share = 100 * height / totals[index]
            if np.isnan(share):
                continue
            y_pos = bottoms[index] + height / 2
            plot_ax.text(
                positions[index],
                y_pos,
                f"{share:.0f}%",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                fontweight="normal",
            )

        bottoms += heights

    plot_ax.set_title(plot_title,fontsize=16)
    plot_ax.set_ylabel(y_label,fontsize=14)
    plot_ax.set_xticks(positions)
    plot_ax.set_xticklabels(frame.index, fontsize=14)
    plot_ax.grid(alpha=0.3, axis="y")
    plot_ax.axhline(0, color="black", linewidth=0.8)


# --------------------------------------------------------------------------------------
# France-only data
# --------------------------------------------------------------------------------------
carrier_colors = {
    "onshorewind": "#4C78A8",
    "solar": "#F2A541",
    "gas": "#E45756",
    "nuclear": "#9D755D",
    "hydro": "#72B7B2",
}

fr_capacity = pd.DataFrame(
    {
        "Base": get_bus_carrier_totals(network_base, "FR", "capacity") / 1000,
        "New": get_bus_carrier_totals(network_new, "FR", "capacity") / 1000,
    }
).T.fillna(0)

fr_generation = pd.DataFrame(
    {
        "Base": get_bus_carrier_totals(network_base, "FR", "generation") / 1e6,
        "New": get_bus_carrier_totals(network_new, "FR", "generation") / 1e6,
    }
).T.fillna(0)

all_carriers = sorted(set(fr_capacity.columns) | set(fr_generation.columns))
fr_capacity = fr_capacity.reindex(columns=all_carriers, fill_value=0)
fr_generation = fr_generation.reindex(columns=all_carriers, fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

plot_stacked_case_comparison(
    axes[0],
    fr_capacity,
    "France Installed Capacity",
    "Capacity (GW)",
    carrier_colors,
)

plot_stacked_case_comparison(
    axes[1],
    fr_generation,
    "France Generation Mix",
    "Generation (TWh)",
    carrier_colors,
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Technology", loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.02),fontsize=14)

plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()




###########################################

# ------------------------------------- II --------------------------------------
# Exports/Imports per country

print("Export Illustrations")


pf.avg_annual_net_export_bar_plot(network_new)
pf.flow_matrix_heatmap(network_new)



#%%
# ------------------------------------- II --------------------------------------
# Line Congestion

print("Line Congestion Illustrations")

# Line loading (how congested each line is)
line_loading = network_new.lines_t.p0.abs() / network_new.lines.s_nom * 100  # in %
line_loading.mean().plot(kind='bar', figsize=(10,5), title='Average Line Loading (%)')
plt.ylabel('Loading (%)')
plt.axhline(100, color='red', linestyle='--', label='Capacity limit')
plt.legend()
plt.show()

# Flow duration curve — how often each line is congested
fig, ax = plt.subplots(figsize=(10, 5), dpi = 300)
for line in network_new.lines.index:
    loading = network_new.lines_t.p0[line].abs() / network_new.lines.s_nom[line] * 100
    sorted_loading = np.sort(loading.values)[::-1]
    ax.plot(sorted_loading, label=line)
ax.axhline(100, color='red', linestyle='--', label='Capacity limit')
ax.set_xlabel('Hours per year (sorted)')
ax.set_ylabel('Line loading (%)')
#ax.set_title('Flow Duration Curve per Line')
ax.legend()
ax.grid(alpha=0.5)
plt.show()


