
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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

