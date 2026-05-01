
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_functions as pf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import cartopy.io.shapereader as shpreader
import seaborn as sns
import os
if os.getlogin() == "magnu": # to fix one member's problems with pypsa's proj_path
    import pyproj
    proj_path = r"C:\Users\magnu\anaconda3\envs\E2Flex\Library\share\proj"
    pyproj.datadir.set_data_dir(proj_path)
import pypsa

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
network_new=network_base.copy()


# ----------------------------------------------------------------------------------------------
#                                 Optimize both networks
# ----------------------------------------------------------------------------------------------
# Optimise base case
network_base.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})

# Fix all capacities from base solution as minimum, but allow expansion
for tech in network_new.generators.index:
    if tech != "nuclear":
        network_new.generators.loc[tech, "p_nom_extendable"] = True
        network_new.generators.loc[tech, "p_nom_min"] = network_base.generators.loc[tech, "p_nom_opt"]

# Remove nuclear and re-optimize
network_new.remove("Generator", "nuclear")
network_new.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0})





#%% ----------------------------------------------------------------------------------------------
#                                 Compare results
### ----------------------------------------------------------------------------------------------

print("------------------------------------------------------------")
print("                       PRICE COMPARISON                     ")


# -------------- Total System Cost-------------------
print("\n=== Total System Cost (M€) ===")
print(f'Basecase: {round(network_base.objective/1e6,2)}')
# Add the annualised capital cost of the nuclear generator in france to the total system cost of the new network
# as we only close the  nuclear generator we have still paid the capital cost of the nuclear generator in france, 
# but we do not have any dispatch cost from the nuclear generator in france
total_cost_new = network_new.objective #+ network_base.generators.loc["nuclear", "capital_cost"] * network_base.generators.loc["nuclear", "p_nom_opt"]
print(f'New case: {round(total_cost_new/1e6,2)}')


# -------------- Nodal Prices -------------------

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


#----------------------------------------------------------------------#
# PRICE PLOT
#------------------------------------------------------------------------#

# Data (ISO3)
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



#%% -------------------------------------------- 
# --------------------------- CAPACITY AND DISPATCH COMPARISON ------------------------

print("----------------------------------------------------------------------------")
print("                    FRENCH CAPACITY AND DISPATCH COMPARISON                 ")


def compare_france(network_base, network_new, country='FR'):
    tech_colors = {
        "onshorewind": "#4C78A8",
        "solar": "#F2A541",
        "OCGT": "#E45756",
        "nuclear": "#9D755D",
        "hydro": "#72B7B2"
    }

    def get_capacity_dispatch(network):
        gens = network.generators[network.generators.bus == country]
        capacity = (gens.p_nom_opt * 1e-3).rename("Capacity (GW)")
        dispatch = (network.generators_t.p[gens.index].sum() * 1e-6).rename("Dispatch (TWh)")
        return pd.DataFrame({"Capacity (GW)": capacity, "Dispatch (TWh)": dispatch})

    df_base = get_capacity_dispatch(network_base)
    df_new  = get_capacity_dispatch(network_new)

    all_techs = df_base.index.union(df_new.index)
    df_base = df_base.reindex(all_techs, fill_value=0)
    df_new  = df_new.reindex(all_techs, fill_value=0)

    colors = [tech_colors.get(t, 'grey') for t in all_techs]

    fig, axes = plt.subplots(1, 2, figsize=(7, 5.), dpi=200)

    for ax, metric in zip(axes, ["Capacity (GW)", "Dispatch (TWh)"]):
        base_vals = df_base[metric]
        new_vals  = df_new[metric]
        base_pct  = base_vals / base_vals.sum() * 100
        new_pct   = new_vals  / new_vals.sum()  * 100

        bottom_base = 0
        bottom_new  = 0
        for tech, color in zip(all_techs, colors):
            for x_pos, vals, pcts, bottom_val in [
                (0, base_vals, base_pct, bottom_base),
                (1, new_vals,  new_pct,  bottom_new)
            ]:
                val = vals[tech]
                pct = pcts[tech]
                if val > 0:
                    ax.bar(x_pos, val, bottom=bottom_val, color=color, width=0.4,
                        alpha=0.8 if x_pos == 1 else 1.0)
                    mid = bottom_val + val / 2
                    if pct >= 8:  # large enough — label inside
                        ax.text(x_pos, mid, f"{pct:.1f}%",
                                ha='center', va='center', fontsize=10,
                                color='white', fontweight='bold')
                    else:  # too small — label outside with line
                        ax.annotate(f"{pct:.1f}%",
                                    xy=(x_pos, mid),
                                    xytext=(x_pos + 0.35, mid),
                                    fontsize=10, ha='left', va='center', color='black',
                                    arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

            bottom_base += base_vals[tech]
            bottom_new  += new_vals[tech]

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Base', 'New'])
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Shared legend
    handles = [plt.Rectangle((0,0),1,1, color=tech_colors.get(t, 'grey')) for t in all_techs]
    fig.legend(handles, all_techs, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(all_techs), frameon=False)

    #plt.suptitle(f"France ({country}): Technology Mix — Base vs. New Case", y=1.08)
    plt.tight_layout()
    plt.show()

compare_france(network_base, network_new)




#%% ---------------------------------- CAPACITIES ACROSS ALL COUNTRIES -----------------------------

print("CAPACITY COMPARISON ACROSS COUNTRIES")

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




#%% ------------------------------------- II --------------------------------------
# Exports/Imports per country

print("----------------------------- Export Illustrations --------------------------")


pf.avg_annual_net_export_bar_plot(network_new)
pf.flow_matrix_heatmap(network_new)



def export_plot_analyses(network_nodes, week = slice('2015-07-10', '2015-07-17')):
    net_export = pd.DataFrame(index=network_nodes.snapshots)
    for line in network_nodes.lines.index:
        bus0 = network_nodes.lines.loc[line, 'bus0']
        bus1 = network_nodes.lines.loc[line, 'bus1']
        flow = network_nodes.lines_t.p0[line]
        net_export[bus0] = net_export.get(bus0, 0) + flow
        net_export[bus1] = net_export.get(bus1, 0) - flow


    # Timeseries of generation per technology per country
    gen_ts_by_bus = {}
    for gen in network_nodes.generators.index:
        bus = network_nodes.generators.loc[gen, 'bus']
        carrier = network_nodes.generators.loc[gen, 'carrier']
        ts = network_nodes.generators_t.p[gen]
        gen_ts_by_bus.setdefault(bus, {})
        if carrier in gen_ts_by_bus[bus]:
            gen_ts_by_bus[bus][carrier] += ts
        else:
            gen_ts_by_bus[bus][carrier] = ts.copy()

    # Correlation between each technology and net export per country
    corr_results = {}
    for country in gen_ts_by_bus.keys():
        corr_results[country] = {}
        for carrier, ts in gen_ts_by_bus[country].items():
            corr = np.corrcoef(ts, net_export[country])[0, 1]
            corr_results[country][carrier] = corr

    # Display as dataframe
    corr_export_df = pd.DataFrame(corr_results).T  # countries as rows, carriers as columns
    #print(corr_export_df)

    print("Pearson Correlation Coefficients between Net Export and Generation")
    fig, ax = plt.subplots(figsize=(14,4))
    sns.heatmap(corr_export_df, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
    plt.show()

    print("Dispatch, Week:", week)
    countries = ['FR', 'CH', 'DE', 'IT', 'BE']
    fig, axes = plt.subplots(len(countries), 1, figsize=(16, 3.3*len(countries)), dpi=300, sharex=True)

    for ax1, code in zip(axes, countries):
        for carrier, ts in gen_ts_by_bus[code].items():
            ax1.plot(ts[week].index, ts[week].values, label=carrier)

        ax1.set_ylabel('Generation (MW)')
        ax1.grid(alpha=0.3)
        ax1.set_title(code)

        ax2 = ax1.twinx()
        ax2.plot(net_export[code][week].index, net_export[code][week].values,
                color='black', linestyle='--', linewidth=1.5, label='Net Export')
        ax2.set_ylabel('Net Export (MW)')
        ax2.axhline(0, color='black', linewidth=0.5)

    axes[-1].set_xlabel('Date')

    # Collect handles from first subplot + net export
    handles, labels = axes[0].get_legend_handles_labels()
    net_export_handle = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Net Export')
    handles.append(net_export_handle)
    labels.append('Net Export')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles), frameon=False)

    plt.suptitle('Generation Dispatch and Net Export (10/07 - 17/07 2015)', y=1.05)
    plt.tight_layout()
    plt.show()


export_plot_analyses(network_base, week = slice('2015-07-10', '2015-07-17'))
export_plot_analyses(network_new, week = slice('2015-07-10', '2015-07-17'))




def export_plot_analyses(network_base, network_new):

    def get_corr_df(network):
        net_export = pd.DataFrame(index=network.snapshots)
        for line in network.lines.index:
            bus0 = network.lines.loc[line, 'bus0']
            bus1 = network.lines.loc[line, 'bus1']
            flow = network.lines_t.p0[line]
            net_export[bus0] = net_export.get(bus0, 0) + flow
            net_export[bus1] = net_export.get(bus1, 0) - flow

        gen_ts_by_bus = {}
        for gen in network.generators.index:
            bus = network.generators.loc[gen, 'bus']
            carrier = network.generators.loc[gen, 'carrier']
            ts = network.generators_t.p[gen]
            gen_ts_by_bus.setdefault(bus, {})
            if carrier in gen_ts_by_bus[bus]:
                gen_ts_by_bus[bus][carrier] += ts
            else:
                gen_ts_by_bus[bus][carrier] = ts.copy()

        corr_results = {}
        for country in gen_ts_by_bus.keys():
            corr_results[country] = {}
            for carrier, ts in gen_ts_by_bus[country].items():
                corr = np.corrcoef(ts, net_export[country])[0, 1]
                corr_results[country][carrier] = corr

        return pd.DataFrame(corr_results).T

    corr_base = get_corr_df(network_base)
    corr_new  = get_corr_df(network_new)

    # Align columns so both have the same technologies
    all_carriers = corr_base.columns.union(corr_new.columns)
    corr_base = corr_base.reindex(columns=all_carriers, fill_value=np.nan)
    corr_new  = corr_new.reindex(columns=all_carriers,  fill_value=np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), dpi=200,
                         gridspec_kw={'hspace': 0.7})

    sns.heatmap(corr_base, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title("Base Case")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(corr_new, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=axes[1], cbar=False)
    axes[1].set_title("New Case")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].tick_params(axis='y', rotation=0)

    for ax in axes:
        ax.set_ylabel("Country")

    # Add horizontal colorbar below both plots
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.03, pad=0.15, shrink=0.3)
    cbar.set_label('Pearson Correlation')

    #plt.suptitle("Correlation: Net Export vs. Generation", y=1.02)
    plt.show()

export_plot_analyses(network_base, network_new)




"""
By inspecting the changes in net export, it can be seen that France remains a net exporter, however,
Germany will export more to France on average than the other way around.
Inspecting the correlation between net export and the dispatch of different technologies across the countries,
it can be seen that France's net export previosuly was positively correlated with nuclear,
meaning that when it increases, export increases, whereas wind had zero correlation,
however, with the added wind capacity, it has a positive correlation,
indicating that it now makes up the majority of its exports.
In both the base and new case, solar has a negative correlation, likely due to the fact
that its neighbouring countries are also subject to Sunny conditions when they exist,
increasing especially the dispatch in Italy - indicated by its correlation between export and solar generation of 0.7-0.8,
and leading to France importing cheap solar energy.
Exports correlation with gas also increases positively across countries, indicating
that gas plays a larger role in dispatch also across borders due to the absence of nuclear
and larger of intermittent renewable sources.

A sample week in July aligns with these observations.

"""






























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


