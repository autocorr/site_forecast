
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from plotly import graph_objects as go

from .. import (CONFIG, KMHOUR_TO_MS, TPW_TO_PWV, VLBA_SITES, VLBA_SITE_NAMES, logger, _now_dir)

COLORS = px.colors.qualitative.G10


def savefig(fig, outname, t_forecast=None, overwrite=True):
    now_dir = _now_dir(t_forecast)
    out_dir = Path(CONFIG.get("Paths", "plots", fallback="./plots")).expanduser() / now_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / outname
    if path.exists() and not overwrite:
        logger.info(f"Figure exists, continuing: {path}")
    else:
        fig.write_html(f"{path}.html", include_plotlyjs="cdn", auto_open=False)
        print_path = Path(*path.parts[-6:])
        logger.info(f"Figure saved: {print_path}")


def get_multi_site_stack_layout(
        n_rows=10,
        v_delta=0.005,
        limits=None,
    ):
    if limits is None:
        limits = (None, None, None, None)
    xmin, xmax, ymin, ymax = limits
    if n_rows < 1:
        raise ValueError(f"Invalid number of sites: {n_rows=}")
    if any(limits) and (xmin > xmax or ymin > ymax):
        raise ValueError(f"Invalid axis limits: {limits=}")
    yaxes_kwargs = {
            f"yaxis{i}": {
                "domain": [(i-1)/n_rows, i/n_rows-v_delta],
                "range": [ymin, ymax],
            }
            for i in range(1, n_rows+1)
    }
    layout = go.Layout(
            legend=dict(traceorder="reversed"),
            height=1000,
            width=1400,
            showlegend=True,
            hoversubplots="axis",
            hovermode="x unified",
            font=dict(size=14),
            xaxis=dict(
                range=[xmin, xmax],
                showspikes=True,
                spikemode="across+marker",
                spikesnap="data",
                spikethickness=1,
                spikedash="dash",
                spikecolor="black",
                minor=dict(
                    ticklen=6,
                    tickcolor="black",
                    tickmode="auto",
                    nticks=6,
                    showgrid=True,

                ),
            ),
            **yaxes_kwargs
    )
    return layout


def get_two_stack_agg_layout(**kwargs):
    layout = get_multi_site_stack_layout(n_rows=2, v_delta=0.01, **kwargs)
    layout.update(hovermode="x")
    return layout


class PlotlyFigureBase:
    def __init__(self, data, limits=None, t_forecast=None):
        if limits is None:
            limits = (None, None, None, None)
        self.data = data
        self.limits = limits
        self.t_forecast = t_forecast
        self.layout = None

    def save(self, outname, **kwargs):
        fig = go.Figure(data=self.data, layout=self.layout)
        savefig(fig, outname, t_forecast=self.t_forecast, **kwargs)
        return fig


class PlotlyFigureVlbaStack(PlotlyFigureBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = get_multi_site_stack_layout(n_rows=10, limits=self.limits)


class PlotlyFigureVlbaAgg(PlotlyFigureBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = get_two_stack_agg_layout(limits=self.limits)


def get_now_trace(x, y_lim, yaxis=None):
    return go.Scatter(
            x=[x, x],
            y=y_lim,
            mode="lines",
            line_width=1.5,
            line_dash="dash",
            line_color="magenta",
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
    )


def get_wind_limit_graph_objects(xmin, xmax, yaxis=None, vlba=True):
    stow_limit = 25 if vlba else 20  # m/s
    labels = ["Q", "A", "K", "U", "X", " !"]
    limits = [  5,   6,   7,  10,  15, stow_limit]  # m/s
    graph_objects = []
    for label, limit in zip(labels, limits):
        if limit == stow_limit:
            color = "black"
            line_width = 2.5
        else:
            color = "dimgray"
            line_width = 1.0
        scatter = go.Scatter(
                x=[xmin, xmax],
                y=[limit, limit],
                mode="lines",
                line_width=line_width,
                line_dash="dot",
                line_color=color,
                showlegend=False,
                yaxis=yaxis,
        )
        graph_objects.append(scatter)
    return graph_objects


def get_annotation_trace(x, y, text, yaxis=None, x_offset=pd.Timedelta("1h")):
    return go.Scatter(
            x=[x+x_offset],
            y=[y],
            mode="text",
            text=[text],
            textposition="top center",
            yaxis=yaxis,
            showlegend=False,
    )


def get_sun_rise_set_by_site(times, station, t_offset=pd.Timedelta("1d")):
    t_center = times.mean()
    t_delta = (times.max() - t_center) + t_offset
    return station.sun_rise_and_sets(t_center, delta=t_delta)


def get_sun_rise_set_patches(rises, sets, y_lim, yaxis="y1", opacity=0.2):
    ymin, ymax = y_lim
    x_vertices = []
    y_vertices = []
    for t_rise, t_set in zip(rises, sets):
        t_rise = t_rise.to_datetime()
        t_set  = t_set.to_datetime()
        x_vertices.extend([t_rise, t_rise, t_set, t_set, t_rise, None])
        y_vertices.extend([ymin, ymax, ymax, ymin, ymin, None])
    return go.Scatter(
            x=x_vertices,
            y=y_vertices,
            fill="toself",
            opacity=opacity,
            line_width=0,
            mode="lines",
            line_color="gray",
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
    )


class VlbaPlotter:
    sites = VLBA_SITES
    site_names = VLBA_SITE_NAMES
    n_sites = len(VLBA_SITE_NAMES)

    def __init__(self, fc):
        self.fc = fc
        self.df = self.fc.weather_ms.df
        self.times = self.df.index.get_level_values("date")
        self.xmin = self.times.min()
        self.xmax = self.times.max()
        self.sun_rise_sets_by_site = {
                site.name: get_sun_rise_set_by_site(self.times, site)
                for site in self.sites
        }

    @property
    def t_forecast(self):
        return self.fc.forecast_time

    @property
    def okay(self):
        return self.fc.weather_ms.okay

    def now_line_trace(self, y_lim, yaxis: str="y1"):
        return get_now_trace(self.t_forecast, y_lim, yaxis)

    def now_line_traces_for_stack(self, y_lim):
        traces = []
        for i, _ in enumerate(self.site_names):
            traces.append(self.now_line_trace(y_lim, yaxis=f"y{i+1}"))
        return traces

    def now_line_traces_for_agg(self, y_lim):
        return [
                self.now_line_trace(y_lim, yaxis="y1"),
                self.now_line_trace(y_lim, yaxis="y2"),
        ]

    def wind_limit_traces(self, yaxis: str="y1"):
        return get_wind_limit_graph_objects(self.xmin, self.xmax, yaxis)

    def wind_limit_traces_for_stack(self):
        traces = []
        for i, _ in enumerate(self.site_names):
            traces.extend(self.wind_limit_traces(yaxis=f"y{i+1}"))
        return traces

    def wind_limit_traces_for_agg(self):
        return [
                *self.wind_limit_traces(yaxis="y1"),
                *self.wind_limit_traces(yaxis="y2"),
        ]

    def site_name_annotations_for_stack(self, y, **kwargs):
        return [
                get_annotation_trace(self.xmin, y, name, yaxis=f"y{i+1}", **kwargs)
                for i, name in enumerate(self.site_names)
        ]

    def sun_rise_set_for_stack(self, y_lim, opacity=0.2):
        traces = []
        for i, (rises, sets) in enumerate(self.sun_rise_sets_by_site.values()):
            traces.append(get_sun_rise_set_patches(rises, sets, y_lim, yaxis=f"y{i+1}", opacity=opacity))
        return traces

    def sun_rise_set_for_agg(self, y_lim, opacity=0.075):
        traces = []
        for rises, sets in self.sun_rise_sets_by_site.values():
            traces.append(get_sun_rise_set_patches(rises, sets, y_lim, yaxis="y1", opacity=opacity))
            traces.append(get_sun_rise_set_patches(rises, sets, y_lim, yaxis="y2", opacity=opacity))
        return traces

    def value_traces(self, col_name, yaxis=None):
        if "wind_speed" in col_name:
            scale_factor = KMHOUR_TO_MS
        else:
            scale_factor = 1
        traces = []
        for i, site in enumerate(self.sites):
            s_df = self.df.xs(site.name, level="site")
            traces.append(go.Scatter(
                    x=s_df.index,
                    y=s_df[col_name] * scale_factor,
                    name=site.name,
                    line=dict(color=COLORS[i]),
                    mode="lines",
                    yaxis=f"y{i+1}" if yaxis is None else yaxis,
            ))
        return traces

    def quantile_traces(self, col_name, yaxis=None):
        if "wind_speed" in col_name:
            scale_factor = KMHOUR_TO_MS
        else:
            scale_factor = 1
        antenna_counts = (3, 4, 5, 6, 7, 9, 10, 8)
        traces = []
        for n_s in antenna_counts:
            q = n_s / 10
            s_df = (
                    self.df[self.df.index.get_level_values("site").isin(self.site_names)]
                    .groupby(level="date")
                    .quantile(min(q, 1.0))
            )
            if n_s == 8:
                line = dict(color="mediumblue", width=4.0)
                marker = dict(size=8)
            else:
                line = dict(color="black", width=1.75)
                marker = dict(size=0.1)
            traces.append(go.Scatter(
                    x=s_df.index,
                    y=s_df.wind_speed_10m * scale_factor,
                    name=f"{q:.1f}",
                    line=line,
                    marker=marker,
                    mode="lines",
                    yaxis=yaxis,
                    showlegend=False,
            ))
        return traces

    def figure_stack(self, ymin, ymax, annotation_offset_scale=0.82):
        data = [
                *self.sun_rise_set_for_stack((ymin, ymax)),
                *self.now_line_traces_for_stack((ymin, ymax)),
                *self.site_name_annotations_for_stack(ymax*annotation_offset_scale),
        ]
        limits = (self.xmin, self.xmax, ymin, ymax)
        return PlotlyFigureVlbaStack(data, limits=limits, t_forecast=self.t_forecast)

    def figure_agg(self, ymin, ymax):
        data = [
                *self.sun_rise_set_for_agg((ymin, ymax)),
                *self.now_line_traces_for_agg((ymin, ymax)),
        ]
        limits = (self.xmin, self.xmax, ymin, ymax)
        return PlotlyFigureVlbaAgg(data, limits=limits, t_forecast=self.t_forecast)

    def plot_stack(self, col_name, ymin, ymax, outname=None):
        if outname is None:
            outname = f"{col_name}_ms_stack"
        fig = self.figure_stack(ymin, ymax)
        if "wind_speed" in col_name:
            fig.data.extend(self.wind_limit_traces_for_stack())
        fig.data.extend(self.value_traces(col_name))
        fig.save(outname=outname)
        return fig

    def plot_agg(self, col_name, ymin, ymax, outname=None):
        if outname is None:
            outname = f"{col_name}_ms_agg"
        fig = self.figure_agg(ymin, ymax)
        if "wind_speed" in col_name:
            fig.data.extend(self.wind_limit_traces_for_agg())
        fig.data.extend([
            *self.value_traces(col_name, yaxis="y1"),
            *self.quantile_traces(col_name, yaxis="y2"),
        ])
        fig.save(outname=outname)
        return fig

    def plot_wind_speed(self, ymin=0, ymax=30) -> None:
        col_name = "wind_speed_10m"
        self.plot_stack(col_name, ymin, ymax, outname="wind_speed_ms_stack")
        self.plot_agg(  col_name, ymin, ymax, outname="wind_speed_ms_agg")

