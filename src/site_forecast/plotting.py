
# TODO
# - wind rose for direction and amplitude
# - resolved cloud cover
# - effective sensitivity

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

from astropy.time import Time
from astropy.coordinates import get_sun, AltAz, EarthLocation

from . import (CONFIG, KMHOUR_TO_MS, TPW_TO_PWV, logger, _now_dir)

VLA_SITE = EarthLocation.of_site("VLA")
CMAP = plt.get_cmap("magma")
CMAP.set_bad("0.85")


warnings.filterwarnings(action="ignore", category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout.*")
warnings.filterwarnings(action="ignore", category=UserWarning,
        message="No artists with labels found to put in legend.*")


def apply_mpl_settings():
    plt.rc("text", usetex=False)
    plt.rc("font", size=10, family="cmu serif")
    plt.rc("mathtext", fontset="cm")
    plt.rc("axes", unicode_minus=False)
    plt.rc("xtick", direction="in", top=True)
    plt.rc("ytick", direction="in", right=True)
    plt.rc("axes", unicode_minus=False)
    plt.ioff()
apply_mpl_settings()


def truncate_colormap(
        cmap_name,
        vmin=0.4,
        vmax=1.0,
        n_color=256,
    ):
    cmap = plt.cm.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
            cmap_name,
            cmap(np.linspace(vmin, vmax, n_color)),
    )


def splice_colormaps(
        cmap1_name,
        cmap2_name,
        pivot=0.5,
        n_color=256,
    ):
    cmap1 = plt.cm.get_cmap(cmap1_name, n_color)
    cmap2 = plt.cm.get_cmap(cmap2_name, n_color)
    pivot_int = int(round(pivot * n_color))
    pivot_int = max(pivot_int, 0)
    pivot_int = min(pivot_int, n_color-1)
    n_cmap1 = pivot_int
    n_cmap2 = n_color-pivot_int
    # splice
    new_colors = cmap1(np.linspace(0, 1, n_color))
    new_colors[:pivot_int,:] = cmap1(np.linspace(0, 1, n_cmap1))
    new_colors[pivot_int:,:] = cmap2(np.linspace(0, 1, n_cmap2))
    return ListedColormap(new_colors)

CMAP_GRAY_MAGMA = splice_colormaps("gray_r", "magma", pivot=0.2)


def colormap_from_herbie_query(hq, cmap_name="magma", split_cmap_name="gray_r"):
    vmin, vmax, split, use_log = hq.plot_min, hq.plot_max, hq.plot_split, hq.plot_log
    if vmax <= vmin:
        raise ValueError(f"Invalid limits: {vmin=}, {vmax=}")
    vmin = hq.data.min().item() if vmin is None else vmin
    vmax = hq.data.max().item() if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    if split is None:
        cmap = getattr(plt.cm, cmap_name)
    else:
        if split < 0 or split > 1:
            raise ValueError(f"Invalid split: {split=}")
        split_frac = split / (vmax - vmin)
        cmap = splice_colormaps(split_cmap_name, cmap_name, split_frac)
    return norm, cmap


def savefig(outname, t_forecast=None, dpi=300, h_pad=0.3, w_pad=None, overwrite=True):
    now_dir = _now_dir(t_forecast)
    out_dir = Path(CONFIG.get("Paths", "plots", fallback="./plots")).expanduser() / now_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / outname
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    if path.exists() and not overwrite:
        logger.info(f"Figure exists, continuing: {path}")
    else:
        for ext in ("pdf", "png"):
            filen = str(path) + f".{ext}"
            plt.savefig(filen, dpi=dpi)
        print_path = Path(*path.parts[-6:])
        logger.info(f"Figure saved: {print_path}")
        plt.close("all")


def set_minor_ticks(ax, x=True, y=True, n_xticks=None, n_yticks=None):
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_xticks))
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_yticks))


def set_color_for_twin(ax, color):
    ax.yaxis.label.set_color(color)
    ax.spines["right"].set_edgecolor(color)
    ax.tick_params(axis="y", colors=color)


def set_grid(ax):
    ax.grid(linestyle="dashed", color="0.3", linewidth=0.3)


def annotate_with_patheffects(
        ax,
        label,
        xy=(0.1, 0.1),
        xycoords="axes fraction",
        linewidth=2,
        color="black",
        foreground="white",
    ):
    anno = ax.annotate(label, xy=xy, xycoords=xycoords, color=color)
    anno.set_path_effects([
        patheffects.withStroke(linewidth=linewidth, foreground=foreground),
    ])
    return anno


def fix_minus_labels(ax, x=False, y=False):
    if x:
        ticks = ax.get_xticks().tolist()[1:-1]
        labels = ax.xaxis.get_ticklabels()[1:-1]
        ax.xaxis.set_ticks(ticks)
        ax.xaxis.set_ticklabels([
                label.get_text().replace(r"\mathdefault", "")
                for label in labels
        ])
    if y:
        ticks = ax.get_yticks().tolist()[1:-1]
        labels = ax.yaxis.get_ticklabels()[1:-1]
        ax.yaxis.set_ticks(ticks)
        ax.yaxis.set_ticklabels([
                label.get_text().replace(r"\mathdefault", "")
                for label in labels
        ])


def add_phase_limits(ax):
    """Overplot atmospheric phase limits for Q, K, U, and X Bands."""
    for limit in (5, 10, 15, 30):  # deg
        ax.axhline(limit, color="deepskyblue", linestyle="dashed", linewidth=1.2, zorder=-2)


def add_wind_limits(ax):
    """
    Overplot wind speed limits for Q, K, U, and X Bands and the stow limit of
    20 m/s.
    """
    for limit in (5, 7, 10, 15):  # m/s
        ax.axhline(limit, color="deepskyblue", linestyle="dashed", linewidth=1.2, zorder=-2)
    ax.axhline(20, color="darkorange", linestyle="dashed", linewidth=1.2, zorder=-2)


def add_night(ax, t_forecast, delta="1d"):
    # TODO Can speed up this portion of code by interpolating the altitudes
    # without having to directly calculate them over short (e.g., ~1 min)
    # intervals.
    dt = pd.Timedelta(delta)
    time = Time(pd.date_range(t_forecast-dt, t_forecast+dt, freq="10min"))
    altaz = AltAz(obstime=time, location=VLA_SITE)
    sun_altaz = get_sun(time).transform_to(altaz)
    sun_up = [co.alt.value > 0 for co in sun_altaz]
    rises, sets = [], []
    if not sun_up[0]:
        sets.append(time[0])
    last = sun_up[0]
    for t, up in zip(time[:-1], sun_up[:-1]):
        if not last and up:
            rises.append(t)
        if last and not up:
            sets.append(t)
        last = up
    if not sun_up[-1]:
        sets.append(time[-1])
    for t_rise, t_set in zip(rises, sets):
        ax.axvspan(t_rise.to_datetime(), t_set.to_datetime(), color="0.85", zorder=-3)


def clip_xlim(ax, t, delta="12h"):
    t = pd.Timestamp(t)
    dt = pd.Timedelta(delta)
    ax.set_xlim(t-dt, t+dt)


def set_dates(ax, minticks=3, maxticks=7):
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = [
            "%y",
            "%b",
            "%d",
            "%H:%M",
            "%H:%M",
            "%S.%f",
    ]
    formatter.zero_formats = [""] + formatter.formats[:-1]
    formatter.zero_formats[3] = "%d %b"
    formatter.offset_formats = [
            "",
            "%Y",
            "%b %Y",
            "%d %b %Y",
            "%d %b %Y",
            "%d %b %Y %H:%M",
    ]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    return locator, formatter


def add_now_line(ax, t_forecast):
    ax.axvline(t_forecast, 0.85, 1.00, color="magenta", linestyle="dotted", zorder=-1)
    ax.axvline(t_forecast, 0.00, 0.15, color="magenta", linestyle="dotted", zorder=-1)
    return ax


def style_single_panel_plot(ax, fc) -> plt.Axes:
    when = fc.forecast_time
    add_night(ax, when)
    add_now_line(ax, when)
    ax.set_xlabel("UTC Time")
    clip_xlim(ax, when)
    set_dates(ax)
    set_grid(ax)
    set_minor_ticks(ax)
    return ax


def draw_all_api_baselines(ax, fc) -> plt.Axes:
    df = fc.phase.df
    cmap = plt.cm.managua
    for i in range(6):
        color = cmap(i/5)
        ax.plot(df.index, df[f"rms_phase{i}"], color=color, label=str(i))
    return ax


def draw_previous_phases(ax, fc) -> plt.Axes:
    previous_forecasts, utc_offsets = fc.get_previous_phase_forecasts()
    n_offsets = len(utc_offsets)
    alpha = 0.3
    for df, offset in zip(previous_forecasts, utc_offsets):
        if df is None:
            continue
        dates = df.index.to_pydatetime()
        phase = df.phase_rms.values
        ax.plot(dates[0], phase[0], linestyle="none", marker=".",
                markerfacecolor="red", markeredgecolor="none", markersize=8,
                alpha=alpha)
        ax.plot(dates[1:], phase[1:], "r-", alpha=alpha)
    return ax


def draw_phase_rms(ax, fc, overplot_previous=True) -> plt.Axes:
    p_df = fc.phase.df
    f_df = fc.predict.df
    if overplot_previous:
        draw_previous_phases(ax, fc)
    if fc.phase.okay:
        ax.plot(p_df.index.to_pydatetime(), p_df.phase_rms, "k-",
                drawstyle="steps-mid")
    if fc.predict.okay:
        ax.plot(f_df.index.to_pydatetime(), f_df.phase_rms, linestyle="solid",
                color="white", linewidth=3)
        ax.plot(f_df.index.to_pydatetime(), f_df.phase_rms, linestyle="solid",
                color="darkred")
    ax.set_xlabel("UTC Time")
    ax.set_ylabel("Phase RMS [deg]")
    ax.set_ylim(0, 46)
    add_phase_limits(ax)
    style_single_panel_plot(ax, fc)
    return ax


def plot_phase_rms_forecast(fc, overplot_previous=True, outname="phase_rms") -> None:
    if not fc.phase.okay and not fc.predict.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    draw_phase_rms(ax, fc, overplot_previous=overplot_previous)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_phase_rms_past(fc, outname="phase_rms_past") -> None:
    if not fc.phase.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    p_df = fc.phase.df
    fig, ax = plt.subplots(figsize=(4, 3))
    draw_all_api_baselines(ax, fc)
    # Phase RMS for OST
    ax.plot(p_df.index, p_df.rms_phase_for_ost, "w-", drawstyle="steps-mid", linewidth=3)
    ax.plot(p_df.index, p_df.rms_phase_for_ost, color="firebrick", drawstyle="steps-mid", label="OST")
    # Median phase RMS
    ax.plot(p_df.index, p_df.phase_rms, "w-", drawstyle="steps-mid", linewidth=3)
    ax.plot(p_df.index, p_df.phase_rms, "k-", drawstyle="steps-mid", label="Med.")
    ax.legend(loc="upper left", fontsize=8, ncols=4, handlelength=1,
            framealpha=0.6, columnspacing=1)
    style_single_panel_plot(ax, fc)
    ax.set_ylabel("Phase RMS [deg]")
    delta = pd.Timedelta("10min")
    ax.set_xlim(
            fc.forecast_time - pd.Timedelta("12h") - delta,
            fc.forecast_time + delta,
    )
    ax.set_ylim(0, 46)
    add_phase_limits(ax)
    savefig(outname, t_forecast=fc.forecast_time)


def draw_wind(ax, fc):
    if fc.station.okay:
        s_df = fc.station.df
        s_dates = s_df.index.to_pydatetime()
        ax.fill_between(s_dates, s_df.wind_speed_minimum,
                s_df.wind_speed_maximum, facecolor="black", edgecolor="none",
                alpha=0.5, step="mid")
        ax.plot(s_dates, s_df.wind_speed_average, color="black",
                drawstyle="steps-mid")
    if fc.weather.okay:
        w_df = fc.weather.df
        w_dates = w_df.index.to_pydatetime()
        ax.fill_between(w_dates, w_df.wind_speed_10m * KMHOUR_TO_MS,
                w_df.wind_gusts_10m * KMHOUR_TO_MS, facecolor="red",
                edgecolor="none", alpha=0.5, step="mid")
        ax.plot(w_dates, w_df.wind_speed_10m * KMHOUR_TO_MS, color="darkred",
                drawstyle="steps-mid")
    ax.set_ylim(0, 25)
    ax.set_ylabel(r"Wind Speed [$\mathrm{m\,s^{-1}}$]")
    add_wind_limits(ax)
    style_single_panel_plot(ax, fc)
    return ax


def plot_wind(fc, outname="wind") -> None:
    if not fc.weather.okay and not fc.station.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    draw_wind(ax, fc)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_temperature(fc, outname="temperature"):
    if not fc.weather.okay and not fc.station.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    if fc.station.okay:
        s_df = fc.station.df
        s_dates = s_df.index.to_pydatetime()
        ax.plot(s_dates, s_df.temperature, "k-", drawstyle="steps-mid")
        ax.plot(s_dates, s_df.dewpoint_temperature, "k--", drawstyle="steps-mid")
    if fc.weather.okay:
        w_df = fc.weather.df
        w_dates = w_df.index.to_pydatetime()
        ax.plot(w_dates, w_df.temperature_2m, color="darkred", drawstyle="steps-mid")
        ax.plot(w_dates, w_df.dew_point_2m, color="darkred", linestyle="dashed", drawstyle="steps-mid")
    ax.set_ylim(-30, 40)
    ax.set_ylabel(r"Temperature [C]")
    style_single_panel_plot(ax, fc)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_direct_radiation(fc, outname="direct_radiation"):
    if not fc.weather.okay and not fc.station.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, ax1 = plt.subplots(figsize=(4, 3))
    if fc.station.okay:
        ax1.plot(fc.station.df.index.to_pydatetime(),
                fc.station.df.pyranometer_2, "k-", drawstyle="steps-mid")
        ax1.set_ylabel("Pyranometer 2 Value")
        ax1.set_ylim(0.22, 6.0)
    ax2 = ax1.twinx()
    if fc.weather.okay:
        # plot direct radiation values
        w_df = fc.weather.df
        w_dates = w_df.index.to_pydatetime()
        ax2.fill_between(w_dates, np.zeros_like(w_df.diffuse_radiation),
                w_df.diffuse_radiation/1e3, facecolor="darkred",
                edgecolor="none", alpha=0.5, step="mid")
        ax2.plot(w_dates, w_df.direct_radiation/1e3, color="darkred", drawstyle="steps-mid")
        ax2.set_ylabel(r"Radiation [$\mathrm{kW\,m^{-2}}$]")
        ax2.set_ylim(-0.1, 1.2)
        set_color_for_twin(ax2, "darkred")
    style_single_panel_plot(ax1, fc)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_pwv(fc, outname="pwv"):
    if not fc.weather.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    w_df = fc.weather.df
    w_dates = w_df.index.to_pydatetime()
    pwv = w_df.total_column_integrated_water_vapour * TPW_TO_PWV  # mm
    ax.plot(w_dates, pwv, color="darkred")
    ax.set_ylim(0, 40)
    ax.set_ylabel("PWV [mm]")
    style_single_panel_plot(ax, fc)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_precipitation(fc, outname="precip"):
    if not fc.weather.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    w_df = fc.weather.df
    w_dates = w_df.index.to_pydatetime()
    fig, axes = plt.subplots(figsize=(4, 4.5), nrows=2, sharex=True)
    ax1, ax2 = axes
    ax1.plot(w_dates, w_df.precipitation_probability, color="darkred")
    ax1.set_ylim(-2.5, 102.5)
    ax1.set_ylabel("Precip. Probability")
    ax2.fill_between(w_dates, w_df.precipitation, step="mid", color="royalblue")
    ax2.set_ylim(-1, 31)
    ax2.set_ylabel("Precip. [mm]")
    for ax in axes:
        style_single_panel_plot(ax, fc)
        ax.label_outer()
    savefig(outname, t_forecast=fc.forecast_time)


def plot_boundary_layer_height(fc, outname="pbl_height"):
    if not fc.weather.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    w_df = fc.weather.df
    w_dates = w_df.index.to_pydatetime()
    fig, ax = plt.subplots(figsize=(4, 3))
    pbl = w_df.boundary_layer_height / 1e3  # km
    ax.plot(w_dates, pbl, color="darkred")
    ax.set_ylim(-0.1, 5.1)
    ax.set_ylabel("Boundary Layer Height [km]")
    style_single_panel_plot(ax, fc)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_cloud_cover_point(fc, outname="cloud_cover_point") -> None:
    if not fc.weather.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, axes = plt.subplots(figsize=(4, 5), nrows=4, sharex=True, sharey=True)
    w_df = fc.weather.df
    w_dates = w_df.index.to_pydatetime()
    zeros = np.zeros(w_dates.shape[0])
    low = w_df.cloud_cover_low  / 100
    mid = w_df.cloud_cover_mid  / 100
    hig = w_df.cloud_cover_high / 100
    tot = w_df.cloud_cover      / 100
    kwargs = dict(edgecolor="black", alpha=0.5)
    axes[0].fill_between(w_dates, zeros, hig, facecolor="skyblue", **kwargs)
    axes[1].fill_between(w_dates, zeros, mid, facecolor="dodgerblue", **kwargs)
    axes[2].fill_between(w_dates, zeros, low, facecolor="darkslateblue", **kwargs)
    axes[3].fill_between(w_dates, zeros, tot, facecolor="black", **kwargs)
    for ax, label in zip(axes, ("High", "Mid", "Low", "Total")):
        style_single_panel_plot(ax, fc)
        ax.set_ylabel(label)
        ax.label_outer()
        ax.set_ylim(-0.05, 1.05)
    savefig(outname, t_forecast=fc.forecast_time)


def plot_herbie_maps(hq, outstem="maps") -> None:
    outname = f"{outstem}_{hq.query_type}"
    if not hq.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    # FIXME Determine correct number of panels and page sizing.
    match hq.n_steps:
        case 49:
            fig, axes = plt.subplots(figsize=(8, 11.3), nrows=8, ncols=6, sharex=True, sharey=True)
        case 13:
            fig, axes = plt.subplots(figsize=(8,  4.3), nrows=3, ncols=6, sharex=True, sharey=True)
        case _:
            raise ValueError(f"Invalid number of steps: {hq.n_steps=}")
    for ax, step in zip(axes.flat, hq.ds.step):
        s_ds = hq.data.sel(step=step)
        data = s_ds.values.copy()
        data[data <= 1e-34] = np.nan
        ax.pcolormesh(s_ds.longitude, s_ds.latitude, data, cmap=CMAP,
                vmin=hq.plot_min, vmax=hq.plot_max, norm=hq.plot_norm_type,
                edgecolors="none", rasterized=True)
        ax.scatter(hq.lon, hq.lat, color="dodgerblue", marker="+")
        ts = pd.Timestamp(s_ds.valid_time.values)
        if step == 0:
            annotate_with_patheffects(ax, f"{hq.label_name}", xy=(0.45, 0.85),
                    color="firebrick")
        if (ts.hour == 0 and ts.minute == 0) or (step == 0):
            annotate_with_patheffects(ax, f"{ts.month_name()[:3]} {ts.day}",
                    xy=(0.05, 0.85), color="firebrick")
        annotate_with_patheffects(ax, f"{ts.hour:0>2d}:{ts.minute:0>2d}", xy=(0.05, 0.05))
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    savefig(outname, t_forecast=hq.forecast_time, h_pad=0.5, w_pad=0.5)


def plot_herbie_cloud_combo(fc, outname="cloud_combo"):
    try:
        mcc = fc.herbie_queries["mcc"]
        tcw = fc.herbie_queries["tcolw"]
        vil = fc.herbie_queries["veril"]
    except KeyError:
        logger.warn(f"Skipping plot: {outname}")
        return
    if not mcc.okay or not tcw.okay or not vil.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    if (n_tcw := tcw.ds.step.shape) != (n_vil := vil.ds.step.shape):
        raise ValueError(f"Different number of timesteps for TCOLW and VIL: {n_tcw=}, {n_vil=}")
    fig, axes = plt.subplots(figsize=(8, 11.3), nrows=8, ncols=6, sharex=True, sharey=True)
    kwargs = {
            "mcc": (
                mcc.ds.interp({"step": vil.ds.step}).mcc,
                [1, 25, 50, 75, 100],
                truncate_colormap("Greys", vmin=0.2, vmax=0.8),
                "linear",
            ),
            "tcw": (
                tcw.ds.tcolw,
                np.logspace(-3, 0, 4),
                plt.cm.get_cmap("Blues"),
                "log",
            ),
            "vil": (
                vil.ds.veril,
                np.logspace(-3, 0, 4),
                plt.cm.get_cmap("Reds"),
                "log",
            ),
    }
    for ax, step in zip(axes.flat, vil.ds.step):
        for kind, (ds, levels, cmap, norm) in kwargs.items():
            s_ds = ds.sel(step=step)
            ax.contourf(s_ds.longitude, s_ds.latitude, s_ds.values,
                    levels=levels, cmap=cmap, extend="max", norm=norm,
                    corner_mask=False, antialiased=True, zorder=2)
            ax.contour(s_ds.longitude, s_ds.latitude, s_ds.values,
                    levels=[levels[0]], colors=[cmap(1.0)], linewidths=1.0,
                    corner_mask=False, antialiased=True, zorder=2)
        ax.scatter(mcc.lon, mcc.lat, color="dodgerblue", marker="+", zorder=100)
        ts = pd.Timestamp(s_ds.valid_time.values)
        if step == 0:
            annotate_with_patheffects(ax, "COMBO", xy=(0.45, 0.85),
                    color="firebrick")
        if (ts.hour == 0 and ts.minute == 0) or (step == 0):
            annotate_with_patheffects(ax, f"{ts.month_name()[:3]} {ts.day}",
                    xy=(0.05, 0.85), color="firebrick")
        annotate_with_patheffects(ax, f"{ts.hour:0>2d}:{ts.minute:0>2d}", xy=(0.05, 0.05))
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    savefig(outname, t_forecast=fc.forecast_time, h_pad=0.5, w_pad=0.15)


def plot_herbie_coverage(hq, thresh=1e-4, outstem="coverage") -> None:
    outname = f"{outstem}_{hq.query_type}"
    if not hq.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    ds = hq.ds
    for radius in ds.radius:
        s_ds = ds.sel(radius=radius)
        label = f"{radius:0>2.1f} km"
        ax.plot(s_ds.date, s_ds[f"{hq.query_type}_c"], drawstyle="steps-mid", label=label)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"{hq.label_name} Coverage")
    style_single_panel_plot(ax, hq)
    ax.set_xlim(ds.date.values.min(), ds.date.values.max())
    savefig(outname, t_forecast=hq.forecast_time)


def draw_herbie_center_point(ax, hq) -> plt.Axes:
    ds = hq.ds
    data = ds[f"{hq.query_type}_p"].values.copy()
    if hq.plot_log:
        data[data == 0] = 1e-34
        data = np.log10(data)
    ax.plot(ds.date, data, "k-", drawstyle="steps-mid", label="Point")
    ax.set_ylabel(hq.label)
    style_single_panel_plot(ax, hq)
    return ax


def draw_herbie_aperture_point(ax, hq, legend=True) -> plt.Axes:
    ds = hq.ds
    for r in ds.radius.values:
        label = f"{r:0>2.1f} km"
        s_ds = ds.sel(radius=r, quantile=0.8)
        data = s_ds[f"{hq.query_type}_q"].values.copy()
        if hq.plot_log:
            data[data == 0] = 1e-34
            data = np.log10(data)
        ax.plot(s_ds.date, data, drawstyle="steps-mid", label=label)
    if legend:
        ax.legend(fontsize=8)
    ax.set_ylabel(hq.label)
    style_single_panel_plot(ax, hq)
    return ax


def clip_herbie_point_limits(ax, hq) -> plt.Axes:
    if hq.plot_min is not None and hq.plot_max is not None:
        if hq.plot_log:
            ax.set_ylim(np.log10(hq.plot_min)-0.2, np.log10(hq.plot_max)+0.2)
        else:
            yrange = hq.plot_max - hq.plot_min
            ylim_min = hq.plot_min - 0.05 * yrange
            ylim_max = hq.plot_max + 0.05 * yrange
            ax.set_ylim(ylim_min, ylim_max)
    return ax


def plot_herbie_point(hq, outstem="point") -> None:
    outname = f"{outstem}_{hq.query_type}"
    if not hq.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    ds = hq.ds
    fig, axes = plt.subplots(figsize=(4, 4), nrows=2, sharex=True, sharey=True)
    ax1, ax2 = axes
    draw_herbie_center_point(ax1, hq)
    draw_herbie_aperture_point(ax2, hq)
    annotate_with_patheffects(ax1, "Point", xy=(0.05, 0.85))
    annotate_with_patheffects(ax2, "Radius, 0.8 quan.", xy=(0.05, 0.85))
    for ax in axes:
        clip_herbie_point_limits(ax, hq)
        ax.label_outer()
        ax.set_xlim(ds.date.min(), ds.date.max())
    savefig(outname, t_forecast=hq.forecast_time)


def plot_herbie_quantile_waterfall(hq, outstem="waterfall") -> None:
    outname = f"{outstem}_{hq.query_type}"
    if not hq.okay:
        logger.warn(f"Skipping plot: {outname}")
        return
    ds = hq.ds
    n_panels = len(ds.radius)
    c_dy = 0.6
    f_dy = 2.0
    tot_y = c_dy + f_dy * n_panels
    height_ratios = [c_dy/tot_y] + n_panels * [f_dy/tot_y]
    fig, axes = plt.subplots(
            figsize=(4, tot_y),
            nrows=1+n_panels,
            height_ratios=height_ratios,
            sharex=True,
            sharey=True,
    )
    axes.flat[0].axis("off")
    for radius, ax in zip(ds.radius, axes.flat[1:]):
        s_ds = hq.ds.sel(radius=radius)
        data = s_ds[f"{hq.query_type}_q"].values.copy()
        data[data <= 1e-34] = 0
        radius_km = f"{radius:0>2.1f} km"
        extent = (
                s_ds.date.values.min(),
                s_ds.date.values.max(),
                s_ds["quantile"].min().item(),
                s_ds["quantile"].max().item(),
        )
        im = ax.imshow(data, aspect="auto", cmap=CMAP, vmin=hq.plot_min,
                vmax=hq.plot_max, norm=hq.plot_norm_type, origin="lower",
                extent=extent, zorder=-10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_ylabel(f"Quantile at {radius_km}")
        set_minor_ticks(ax)
        set_grid(ax)
        add_now_line(ax, hq.forecast_time)
    set_dates(ax)
    ax.set_xlabel("UTC Time")
    # Add axes takes dimensions: (left, bottom, width, height)
    ax_colorbar = fig.add_axes([0.25, 1-0.5*c_dy/tot_y, 0.6, 0.3*c_dy/tot_y])
    cb = fig.colorbar(im, cax=ax_colorbar, orientation="horizontal", extend="max")
    cb.set_label(hq.label)
    cb.ax.xaxis.labelpad = 1
    if hq.plot_split is not None:
        ax_colorbar.axvline(hq.plot_split, color="white", linestyle="dotted", linewidth=1.0)
    if hq.plot_log:
        fix_minus_labels(cb.ax, x=True)
    savefig(outname, t_forecast=hq.forecast_time)


def draw_operator_cloud_series(ax, fc) -> plt.Axes:
    colors = {"mcc": "0.3", "tcolw": "dodgerblue", "veril": "firebrick"}
    for label, hq in fc.herbie_queries.items():
        if not hq.okay:
            logger.warn(f"Skipping plot '{outname}' panel '{label}'")
            continue
        ds = hq.ds
        s_ds = ds.sel(radius=10.0)
        ax.plot(s_ds.date, s_ds[f"{hq.query_type}_c"], color=colors[label],
                drawstyle="steps-mid", label=label.upper())
    style_single_panel_plot(ax, fc)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"Cloud Coverage")
    ax.legend(loc="lower right", fontsize=8, handlelength=1, framealpha=0.6)
    return ax


def draw_band_limit_strip(ax, fc) -> plt.Axes:
    # FIXME This code will be fairly brittle because it relies on indexing into
    # each of the respective quantities, and there's currently no error-checking
    # on if the data products are well-formed. This should be refactored into
    # a specific class that evaluates band limits from the values, but in the
    # mean-time, just throw this into a try/except, since we're only using
    # this for plotting at the current time and it's okay if it doesn't plot.
    # Limits for:  [ Q,  A,  K,  U, >X]
    wind_limits  = [ 5,  6,  7, 10, 15]  # m/s
    phase_limits = [ 5,  7, 10, 15, 30]  # deg
    selfc_limits = [10, 14, 20, 30, 60]  # deg
    try:
        # Get the time axis for (0, +8) hr
        when = fc.forecast_time.round("15min")
        time = pd.date_range(when, when+pd.Timedelta("9h"), freq="15min")
        band = np.zeros(len(time), dtype=int)
        p_df = fc.predict.df["phase_rms"]
        w_df = fc.weather.df["wind_speed_10m"]
        c_df = fc.herbie_queries["tcolw"].ds.sel(quantile=0.8, radius=10.0).tcolw_q.to_dataframe()["tcolw_q"]
        c_df.index = c_df.index.tz_localize("utc")
        for i_t, t in enumerate(time):
            this_phase = p_df.loc[t]
            this_wind  = w_df.loc[t] * KMHOUR_TO_MS
            this_cloud = c_df.loc[t]
            for i_l, (w_limit, p_limit) in enumerate(zip(wind_limits, phase_limits)):
                if this_wind > w_limit or this_phase > p_limit:
                    band[i_t] = i_l
                if this_cloud > 1e-1:
                    band[i_t] = 4  # >X
        # Plot values
        cmap = ListedColormap(
                ["darkorchid", "royalblue", "mediumturquoise", "gold", "0.5"],
        )
        ax.imshow(band.reshape((1, -1)), cmap=cmap, vmin=0, vmax=4,
                aspect="auto", extent=[time.min(), time.max(), -0.4, 0.4])
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticks([])
    except:
        logger.warn("Could not draw band limits in operator summary.")
    return ax


def plot_operator_summary(fc, outname="summary") -> None:
    """
    Plot a summary of (a) phase RMS, (b) wind speed, (c) cloud coverage for
    MCC, TCOLW, and VERIL.
    """
    fig, axes = plt.subplots(figsize=(5, 7.5), nrows=4, sharex=True,
            height_ratios=[0.1, 1, 1, 1])
    ax1, ax2, ax3, ax4 = axes
    draw_band_limit_strip(ax1, fc)
    draw_phase_rms(ax2, fc)
    draw_wind(ax3, fc)
    draw_operator_cloud_series(ax4, fc)
    for ax in axes:
        ax.label_outer()
    ax4.set_xlim(
            fc.forecast_time+pd.Timedelta("-1.5h"),
            fc.forecast_time+pd.Timedelta("+9.0h"),
    )
    savefig(outname, t_forecast=fc.forecast_time, h_pad=0.15)


def plot_all_weather(fc):
    plot_phase_rms_forecast(fc)
    plot_phase_rms_past(fc)
    plot_wind(fc)
    plot_temperature(fc)
    plot_direct_radiation(fc)
    plot_pwv(fc)
    plot_precipitation(fc)
    plot_boundary_layer_height(fc)
    plot_cloud_cover_point(fc)
    plot_operator_summary(fc)
    for hq in fc.herbie_queries.values():
        if not hq.okay:
            logger.warn(f"Skipping plots for: {hq.query_type}")
            return
        plot_herbie_maps(hq)
        plot_herbie_point(hq)
        plot_herbie_coverage(hq)
        plot_herbie_quantile_waterfall(hq)
    plot_herbie_cloud_combo(fc)


def plot_backtest(
        target,
        backtest,
        outname="backtest",
        t_min=datetime(2024, 4, 1),
        t_max=datetime(2024, 6, 1),
    ):
    fig, ax = plt.subplots(figsize=(8, 3))
    add_phase_limits(ax)
    for i, (t, b) in enumerate(zip(target, backtest)):
        if i == 0:
            t.plot(color="black", linewidth=0.75, label="Measured")
            b.plot(color="red", linewidth=0.75, label="Forecast")
        else:
            t.plot(color="black", linewidth=0.75, label=None)
            b.plot(color="red", linewidth=0.75, label=None)
    set_grid(ax)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 46)
    ax.set_ylabel("Phase RMS [deg]")
    savefig(outname)

