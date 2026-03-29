
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns


def plot_first_hour_trade_and_flow(
    network,
    t0=None,
    extent=(-5, 15, 40, 55),
    show_demand_generation=True,
):
    """Plot cross-border flow map for one snapshot.

    Args:
        network: PyPSA network with solved dispatch and line flows.
        t0: Snapshot to plot. If None, uses the first snapshot.
        extent: Map extent as (lon_min, lon_max, lat_min, lat_max).
        show_demand_generation: If True, add D/G labels for each bus.
    """
    if t0 is None:
        t0 = network.snapshots[0]

    # Flow map for selected hour.
    flows_t0 = network.lines_t.p0.loc[t0]
    max_abs_flow = max(1.0, float(flows_t0.abs().max()))
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=0.0, vmax=max_abs_flow)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff")
    ax.add_feature(cfeature.LAND, facecolor="#f7f7f2")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, alpha=0.7)

    network.plot(
        ax=ax,
        margin=0.2,
        bus_sizes=0.02,
        bus_colors="#8582c7",
        bus_alpha=0.9,
        line_colors="lightgray",
        line_widths=1.2,
        title=f"Cross-border Flows at {pd.Timestamp(t0).strftime('%Y-%m-%d %H:%M')}",
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if show_demand_generation:
        gen_t0_by_bus = network.generators_t.p.loc[t0].groupby(network.generators.bus).sum()
        load_t0_by_bus = network.loads_t.p.loc[t0].groupby(network.loads.bus).sum()

    for line in network.lines.index:
        bus0 = network.lines.loc[line, "bus0"]
        bus1 = network.lines.loc[line, "bus1"]
        flow = float(flows_t0[line])

        x0, y0 = network.buses.loc[bus0, ["x", "y"]]
        x1, y1 = network.buses.loc[bus1, ["x", "y"]]

        color = cmap(norm(abs(flow)))
        lw = 1.5 + 4.0 * abs(flow) / max_abs_flow
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=lw,
            alpha=0.95,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        vx = x1 - x0
        vy = y1 - y0
        seg_len = max(1e-9, float(np.hypot(vx, vy)))
        nx = -vy / seg_len
        ny = vx / seg_len

        if abs(flow) > 1:
            if flow >= 0:
                xa, ya, xb, yb = x0, y0, x1, y1
            else:
                xa, ya, xb, yb = x1, y1, x0, y0

            x_start = xa + 0.62 * (xb - xa)
            y_start = ya + 0.62 * (yb - ya)
            dx = 0.14 * (xb - xa)
            dy = 0.14 * (yb - ya)

            ax.annotate(
                "",
                xy=(x_start + dx, y_start + dy),
                xytext=(x_start, y_start),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1f1f1f",
                    lw=1.4,
                    mutation_scale=13,
                    alpha=0.95,
                ),
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

        label_offset = 0.22
        xt = (x0 + x1) / 2 + label_offset * nx
        yt = (y0 + y1) / 2 + label_offset * ny
        ax.text(
            xt,
            yt,
            f"{abs(flow):.0f} MW",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, linewidth=0),
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    for bus in network.buses.index:
        x = network.buses.loc[bus, "x"]
        y = network.buses.loc[bus, "y"]

        ax.text(
            x + 0.15,
            y + 0.15,
            bus,
            fontsize=9,
            weight="bold",
            color="#1f1f1f",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        if show_demand_generation:
            generation = float(gen_t0_by_bus.get(bus, 0.0))
            demand = float(load_t0_by_bus.get(bus, 0.0))
            ax.text(
                x + 0.15,
                y - 0.22,
                f"D: {demand:.0f} MW\nG: {generation:.0f} MW",
                fontsize=7.5,
                ha="left",
                va="top",
                color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, linewidth=0),
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Power Flow (MW)")
    tick_vals = np.linspace(0.0, max_abs_flow, 6)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{v:.0f}" for v in tick_vals])

    plt.tight_layout()
    plt.show()


def interannual_var_boxplots(df_results, figsize = (8,7), dpi = 300):
    # Set professional style
    sns.set(style="whitegrid", context="talk")

    # Define professional color palettes
    tech_colors = {
        'OCGT': '#4c72b0',      # muted blue
        'onshorewind': '#55a868', # muted green
        'solar': '#c44e52',        # muted red
        'nuclear': '#8172b3'    # muted purple
    }

    year_palette = sns.color_palette("coolwarm", n_colors=df_results['year'].nunique())

    plt.figure(figsize=figsize, dpi=dpi)

    # Boxplot for distribution
    ax = sns.boxplot(
        data=df_results,
        x='generator',
        y='p_nom',
        palette=tech_colors,
        showfliers=False,
        width=0.6,
        boxprops=dict(alpha=0.6)
    )

    # Swarmplot for individual years
    sns.swarmplot(
        data=df_results,
        x='generator',
        y='p_nom',
        hue='year',
        palette=year_palette,
        size=11,
        dodge=True,
        alpha=0.85,
        ax=ax
    )

    plt.ylabel("Optimal Capacity (MW)")
    plt.xlabel("Generator")
    #plt.title("Optimal Capacity Across Weather Years")

    # Rotate x-ticks for readability
    plt.xticks(rotation=30)

    # Place legend neatly outside
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title="Year", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False)

    plt.tight_layout()
    plt.show()


def weekly_dispatch_plot(network, tech_colors, start_day, storage = False, figsize = (15,5), dpi = 300):
    # Plot dispatch (In winter, first week)
    plot_time=24*7
    plot_delay = start_day
    # Plot dispatch (In summer)
    plt.figure(figsize=figsize, dpi = dpi)
    plt.plot(network.loads_t.p['load'][plot_delay:plot_delay+plot_time], color=tech_colors['demand'], label='demand')
    plt.plot(network.generators_t.p['onshorewind'][plot_delay:plot_delay+plot_time], color=tech_colors['onshorewind'], label='onshore wind')
    plt.plot(network.generators_t.p['solar'][plot_delay:plot_delay+plot_time], color=tech_colors['solar'], label='solar')
    plt.plot(network.generators_t.p['OCGT'][plot_delay:plot_delay+plot_time], color=tech_colors['OCGT'], label='gas (OCGT)')
    plt.plot(network.generators_t.p['nuclear'][plot_delay:plot_delay+plot_time], color=tech_colors['nuclear'], label='Nuclear')

    if storage:
        # Plot Storage Dispatch (Positive is discharging, Negative is charging)
        plt.fill_between(network.snapshots[plot_delay:plot_delay+plot_time], 
                        network.storage_units_t.p[plot_delay:plot_delay+plot_time]["Pumped Hydro"], 
                        color='teal', label='Storage Dispatch', alpha=0.5)

        plt.legend(loc='upper right')
        plt.title('System Dispatch with Pumped Hydro Storage (One Week)')
        plt.ylabel('Power (MW)')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        plt.legend(fancybox=True, ncol=3,shadow=True, loc='best')
        plt.title('Electricity Dispatch (In summer)')
        plt.xlabel('Hour')
        plt.ylabel('Power (MW)')
        plt.grid(alpha=0.3)
        plt.show()


def energy_mix_piechart(network, colors, labels, full_year = True, dpi = 300):
    if full_year:
        # Pie chart for energy mix
        labels = ['Onshore Wind', 'Solar', 'Gas (OCGT)', 'Nuclear']
        sizes = [float(network.generators_t.p['onshorewind'].sum()),
                float(network.generators_t.p['solar'].sum()),
                float(network.generators_t.p['OCGT'].sum()),
                float(network.generators_t.p['nuclear'].sum())]
        plt.figure(dpi = 300)
        plt.pie(sizes, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
        plt.axis('equal')
        plt.title('Electricity Mix (Full year)', y=1.07)
        plt.show()
    else:
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

        # --- Winter electricity mix ---
        plt.figure(dpi = dpi)
        plt.pie(winter_sizes, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
        plt.axis('equal')
        plt.title('Electricity Mix (Winter)', y=1.07)
        plt.show()

        # --- Summer electricity mix ---
        plt.figure(dpi = dpi)
        plt.pie(summer_sizes, colors=colors, labels=labels, wedgeprops={'linewidth': 0})
        plt.axis('equal')
        plt.title('Electricity Mix (Summer)', y=1.07)
        plt.show()


def duration_curves(network, tech_colors, figsize = (12,8), dpi = 400):
    plt.figure(figsize=figsize, dpi = dpi)

    for gen in network.generators_t.p.columns:
        sorted_values = np.sort(network.generators_t.p[gen].values)[::-1]  # sort descending
        hours = np.arange(1, len(sorted_values)+1)
        plt.plot(hours, sorted_values, label=gen, color=tech_colors.get(gen, 'gray'))

    plt.xlabel("Hours per year (sorted)", fontsize = 14)
    plt.ylabel("Power (MW)", fontsize = 14)
    #plt.title("Annual Generation Duration Curve")
    plt.legend(fancybox=True, shadow=True, ncol = 4, loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()


def annual_capacities_and_dispatch(df_results, tech_colors):
    # Average optimal capacity per year
    plt.figure(figsize=(10,5), dpi = 300)
    sns.barplot(data=df_results, x='year', y='p_nom', hue='generator', palette=tech_colors)
    plt.ylabel("Optimal Capacity (MW)")
    plt.title("Optimal Generator Capacities for Different Weather Years")
    plt.show()

    # Variability of dispatch per year
    plt.figure(figsize=(10,5), dpi = 300)
    sns.barplot(data=df_results, x='year', y='dispatch_std', hue='generator', palette=tech_colors)
    plt.ylabel("Dispatch Standard Deviation (MW)")
    plt.title("Generator Dispatch Variability for Different Weather Years")
    plt.show()

    # Stacked Bar Chart: Annual Dispatch per Technology per Year

    # Sum total dispatch per year and per generator
    dispatch_per_year = df_results.groupby(['year', 'generator'])['total_dispatch'].sum().unstack().fillna(0)

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


def weekly_soc_plot(network_storage, start_day):
    delay = start_day
    plot_time = 24*7

    plt.figure(figsize=(12, 4))
    plt.plot(network_storage.storage_units_t.state_of_charge[delay:delay+plot_time]["Pumped Hydro"], color='teal')
    plt.title('Pumped Hydro State of Charge (MWh)')
    plt.ylabel('Energy Stored (MWh)')
    plt.fill_between(network_storage.snapshots[delay:delay+plot_time], 
                    network_storage.storage_units_t.state_of_charge[delay:delay+plot_time]["Pumped Hydro"], 
                    color='teal', alpha=0.2)
    plt.grid(True, alpha=0.3)
    plt.show()