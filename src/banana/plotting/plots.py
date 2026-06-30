"""Plotting functions for BLI/SPR data and kinetic fits."""

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from banana.io.titration import TitrationSeries
from banana.fitting.fit import FitResult
from banana.models.kinetics import PiecewiseExponential


def _param_stderrs(res: FitResult) -> List[Optional[float]]:
    """Per-parameter standard errors from diagonal of covariance matrix."""
    n = len(res.params)
    out: List[Optional[float]] = [None] * n
    if res.cov is None or res.cov.size == 0:
        return out
    for i in range(min(n, res.cov.shape[0])):
        try:
            out[i] = float(np.sqrt(np.abs(res.cov[i, i])))
        except Exception:
            out[i] = None
    return out


def _fmt_pm(val: float, err: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if err is None or (isinstance(err, float) and (np.isnan(err) or err <= 0)):
        return f"{val:.4g}"
    return f"{val:.4g} ± {err:.3g}"


def _annotate_axes_fit_params(
    ax,
    res: FitResult,
    model_type: str = "piecewise",
    fontsize: float = 7.0,
):
    """Add a text box with fitted parameters ± stderr (and fixed A0, t1, t2 for piecewise)."""
    if model_type != "piecewise":
        return
    lines: List[str] = []
    if res.success and len(res.params) >= 4 and len(res.param_names) >= 4:
        sts = _param_stderrs(res)
        for j in range(4):
            name = res.param_names[j]
            v = float(res.params[j])
            se = sts[j] if j < len(sts) else None
            if se is not None and not (isinstance(se, float) and np.isnan(se)):
                lines.append(f"{name} = {v:.4g} ± {se:.3g}")
            else:
                lines.append(f"{name} = {v:.4g}")
        extra = res.extra or {}
        for key, disp in (("A0", "A0"), ("t1", "t1 (assoc start)"), ("t2", "t2 (dissoc start)")):
            if key in extra:
                lines.append(f"{disp} (fixed) = {float(extra[key]):.6g}")
        if res.chi2 is not None:
            lines.append(f"χ² = {float(res.chi2):.4g}")
    if not lines:
        return

    text = "\n".join(lines)
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
        family="monospace",
        zorder=5,
    )


def _x_axis_for_trace(titration: TitrationSeries, trace_index: int, t: np.ndarray):
    x_plot = t
    using_assoc_time = False
    if (
        getattr(titration, "time_from_assoc_start", None) is not None
        and trace_index < len(titration.time_from_assoc_start)
        and len(titration.time_from_assoc_start[trace_index]) == len(t)
    ):
        x_plot = titration.time_from_assoc_start[trace_index]
        using_assoc_time = True
    return x_plot, using_assoc_time


def _fit_curve_for_trace(
    t: np.ndarray,
    r: np.ndarray,
    res: FitResult,
    model_type: str,
):
    if not (res.success and model_type == "piecewise" and len(res.params) >= 4):
        return None
    try:
        extra = res.extra or {}
        A0 = extra.get(
            "A0",
            float(np.mean(r[: min(50, max(1, len(r) // 10))])),
        )
        t1 = extra.get("t1", float(t[0]))
        t2 = extra.get("t2", float(t[0]) + 0.6 * (float(t[-1]) - float(t[0])))
        model = PiecewiseExponential(A0=A0, t1=t1, t2=t2)
        return model(t, *res.params[:4])
    except Exception:
        return None


def plot_raw_data(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None,
    time_col: str = "Time (sec)",
    response_col: str = "Binding (nm)",
    sensor_col: str = "Sensor",
    ax=None,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if sensor_col in df.columns and sensors is None:
        sensors = df[sensor_col].unique().tolist()

    if sensors:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(sensors), 1)))
        for i, sens in enumerate(sensors):
            subset = df[df[sensor_col] == sens]
            ax.plot(
                subset[time_col],
                subset[response_col],
                label=sens,
                color=colors[i % len(colors)],
                **kwargs,
            )
    else:
        ax.plot(df[time_col], df[response_col], **kwargs)

    ax.set_xlabel(time_col)
    ax.set_ylabel(response_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_titration_fits(
    titration: TitrationSeries,
    trace_results: Optional[List[FitResult]] = None,
    model_type: str = "piecewise",
    figsize: tuple = (12, 8),
    ncols: int = 2,
    plot_data_only: bool = False,
    annotate_fit_params: bool = True,
    **kwargs,
):
    """
    Plot titration traces. Always plots data. Overlays fit only when successful
    unless plot_data_only=True.
    """
    n = len(titration)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No titration traces", ha="center", va="center", transform=ax.transAxes)
        return fig

    trace_results = trace_results or []
    while len(trace_results) < n:
        trace_results.append(
            FitResult(
                success=False,
                params=np.array([]),
                param_names=[],
                cov=None,
                residuals=None,
                chi2=None,
                message="no fit",
            )
        )

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue

        t, r = titration.get_trace(i)
        x_plot, using_assoc_time = _x_axis_for_trace(titration, i, t)
        conc = titration.concentration[i]
        label = titration.labels[i] if titration.labels else f"{conc} nM"

        ax.plot(x_plot, r, "o", markersize=2, alpha=0.6, label="Data", color="C0")

        res = trace_results[i]
        fit_curve = None if plot_data_only else _fit_curve_for_trace(t, r, res, model_type)
        fit_drawn = fit_curve is not None
        if fit_drawn:
            ax.plot(x_plot, fit_curve, "-", linewidth=2, label="Fit", color="C1")

        if not fit_drawn and not plot_data_only:
            ax.text(
                0.02,
                0.98,
                "Fit failed / skipped",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                color="red",
                alpha=0.9,
            )
        elif plot_data_only:
            ax.text(
                0.02,
                0.98,
                "Data only",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                alpha=0.7,
            )

        ax.set_title(f"{label} ({conc} nM)")
        ax.set_xlabel("Time from association start (sec)" if using_assoc_time else "Time (sec)")
        ax.set_ylabel("Response")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if annotate_fit_params and not plot_data_only and fit_drawn:
            _annotate_axes_fit_params(ax, res, model_type=model_type)

    plt.tight_layout()
    return fig


def plot_titration_trace_detailed_page(
    titration: TitrationSeries,
    trace_results: Optional[List[FitResult]],
    trace_index: int,
    raw_df: Optional[pd.DataFrame] = None,
    ligand_sensor: Optional[str] = None,
    reference_sensor: Optional[str] = None,
    model_type: str = "piecewise",
    figsize: tuple = (8.5, 11.0),
    annotate_fit_params: bool = True,
):
    """Single-page detailed layout for one concentration trace."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.0], hspace=0.4, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, :])
    ax_assoc = fig.add_subplot(gs[1, 0])
    ax_dissoc = fig.add_subplot(gs[1, 1])
    ax_raw = fig.add_subplot(gs[2, 0])
    ax_resid = fig.add_subplot(gs[2, 1])

    t, r = titration.get_trace(trace_index)
    x_plot, using_assoc_time = _x_axis_for_trace(titration, trace_index, t)
    conc = titration.concentration[trace_index]
    label = titration.labels[trace_index] if titration.labels else f"{conc} nM"
    res = (
        trace_results[trace_index]
        if trace_results and trace_index < len(trace_results)
        else FitResult(
            success=False,
            params=np.array([]),
            param_names=[],
            cov=None,
            residuals=None,
            chi2=None,
            message="no fit",
        )
    )

    # Top panel: processed trace + fit (same core content as single_plot).
    ax_top.plot(x_plot, r, "o", markersize=2, alpha=0.6, label="Data", color="C0")
    fit_curve = _fit_curve_for_trace(t, r, res, model_type)
    if fit_curve is not None:
        ax_top.plot(x_plot, fit_curve, "-", linewidth=2, label="Fit", color="C1")
    else:
        ax_top.text(
            0.02,
            0.98,
            "Fit failed / skipped",
            transform=ax_top.transAxes,
            fontsize=8,
            verticalalignment="top",
            color="red",
            alpha=0.9,
        )
    ax_top.set_title(f"{label} ({conc} nM)")
    ax_top.set_xlabel("Time from association start (sec)" if using_assoc_time else "Time (sec)")
    ax_top.set_ylabel("Response")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(True, alpha=0.3)
    if annotate_fit_params and fit_curve is not None:
        _annotate_axes_fit_params(ax_top, res, model_type=model_type)

    # Middle row: zoomed phase views from the top panel.
    assoc_mask = None
    if (
        getattr(titration, "assoc_mask", None) is not None
        and trace_index < len(titration.assoc_mask)
        and len(titration.assoc_mask[trace_index]) == len(t)
    ):
        assoc_mask = np.asarray(titration.assoc_mask[trace_index], dtype=bool)
    dissoc_mask = None
    if (
        getattr(titration, "dissoc_mask", None) is not None
        and trace_index < len(titration.dissoc_mask)
        and len(titration.dissoc_mask[trace_index]) == len(t)
    ):
        dissoc_mask = np.asarray(titration.dissoc_mask[trace_index], dtype=bool)

    if assoc_mask is None:
        assoc_mask = np.asarray(x_plot, dtype=float) >= 0.0
    if dissoc_mask is None:
        dissoc_mask = ~assoc_mask

    if np.any(assoc_mask):
        ax_assoc.plot(np.asarray(x_plot)[assoc_mask], np.asarray(r)[assoc_mask], "o", markersize=2, alpha=0.6, color="C0", label="Data")
        if fit_curve is not None:
            ax_assoc.plot(np.asarray(x_plot)[assoc_mask], np.asarray(fit_curve)[assoc_mask], "-", linewidth=1.5, color="C1", label="Fit")
    else:
        ax_assoc.text(0.5, 0.5, "Association phase unavailable", ha="center", va="center", transform=ax_assoc.transAxes)
    ax_assoc.set_title("Association phase (zoom)")
    ax_assoc.set_xlabel("Time from association start (sec)" if using_assoc_time else "Time (sec)")
    ax_assoc.set_ylabel("Response")
    if np.any(assoc_mask):
        ax_assoc.legend(loc="best", fontsize=8)
    ax_assoc.grid(True, alpha=0.3)

    if np.any(dissoc_mask):
        ax_dissoc.plot(np.asarray(x_plot)[dissoc_mask], np.asarray(r)[dissoc_mask], "o", markersize=2, alpha=0.6, color="C0", label="Data")
        if fit_curve is not None:
            ax_dissoc.plot(np.asarray(x_plot)[dissoc_mask], np.asarray(fit_curve)[dissoc_mask], "-", linewidth=1.5, color="C1", label="Fit")
    else:
        ax_dissoc.text(0.5, 0.5, "Dissociation phase unavailable", ha="center", va="center", transform=ax_dissoc.transAxes)
    ax_dissoc.set_title("Dissociation phase (zoom)")
    ax_dissoc.set_xlabel("Time from association start (sec)" if using_assoc_time else "Time (sec)")
    ax_dissoc.set_ylabel("Response")
    if np.any(dissoc_mask):
        ax_dissoc.legend(loc="best", fontsize=8)
    ax_dissoc.grid(True, alpha=0.3)

    # Bottom-left: raw ligand/reference traces baseline-adjusted to t=0.
    raw_plotted = False
    if raw_df is not None and "Sensor" in raw_df.columns and "Time (sec)" in raw_df.columns:
        assoc_start_abs = float(t[0] - x_plot[0]) if len(t) > 0 and len(x_plot) > 0 else 0.0

        if ligand_sensor:
            lig = raw_df[raw_df["Sensor"] == ligand_sensor].copy()
            if not lig.empty:
                lig = lig.sort_values("Time (sec)")
                lig_t = lig["Time (sec)"].to_numpy(dtype=float) - assoc_start_abs
                lig_y = lig["Binding (nm)"].to_numpy(dtype=float)
                if lig_y.size:
                    lig_y = lig_y - float(lig_y[0])
                    ax_raw.plot(lig_t, lig_y, "-", linewidth=1.2, label=f"{ligand_sensor} raw")
                    raw_plotted = True

        if reference_sensor:
            ref = raw_df[raw_df["Sensor"] == reference_sensor].copy()
            if not ref.empty:
                ref = ref.sort_values("Time (sec)")
                ref_t = ref["Time (sec)"].to_numpy(dtype=float) - assoc_start_abs
                ref_y = ref["Binding (nm)"].to_numpy(dtype=float)
                if ref_y.size:
                    ref_y = ref_y - float(ref_y[0])
                    ax_raw.plot(ref_t, ref_y, "-", linewidth=1.2, label=f"{reference_sensor} raw")
                    raw_plotted = True

    if not raw_plotted:
        ax_raw.text(0.5, 0.5, "Raw trace(s) unavailable", ha="center", va="center", transform=ax_raw.transAxes)
    ax_raw.set_title("Raw ligand/reference (baseline at t=0)")
    ax_raw.set_xlabel("Time from association start (sec)" if using_assoc_time else "Time (sec)")
    ax_raw.set_ylabel("Binding (nm), baseline-adjusted")
    if raw_plotted:
        ax_raw.legend(loc="best", fontsize=8)
    ax_raw.grid(True, alpha=0.3)

    # Bottom-right: residuals of fit, if available.
    if fit_curve is not None:
        residuals = np.asarray(r, dtype=float) - np.asarray(fit_curve, dtype=float)
        ax_resid.plot(x_plot, residuals, "-", linewidth=1.2, color="C3", label="Residuals")
        ax_resid.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax_resid.legend(loc="best", fontsize=8)
    else:
        ax_resid.text(0.5, 0.5, "Residuals unavailable (no fit)", ha="center", va="center", transform=ax_resid.transAxes)
    ax_resid.set_title("Fit residuals")
    ax_resid.set_xlabel("Time from association start (sec)" if using_assoc_time else "Time (sec)")
    ax_resid.set_ylabel("Data - Fit")
    ax_resid.grid(True, alpha=0.3)

    return fig


def plot_titration_fits_summary_table_page(
    titration: TitrationSeries,
    trace_results: List[FitResult],
    global_result: FitResult,
    model_type: str = "piecewise",
    figsize: tuple = (8.5, 11.0),
):
    """
    One PDF page: table of per-trace fitted parameters (with stderr) and global k_on/k_off/K_D.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.22)
    ax_tbl = fig.add_subplot(gs[0])
    ax_tbl.axis("off")
    ax_tbl.set_title("Per-concentration piecewise fit parameters", fontsize=11, fontweight="bold", pad=8)

    n = len(titration)
    rows: List[List[str]] = []
    col_labels = [
        "Conc (nM)",
        "Label",
        "A1 ± σ",
        "τ1 ± σ",
        "A2 ± σ",
        "τ2 ± σ",
        "χ²",
    ]

    tr_list = list(trace_results or [])
    while len(tr_list) < n:
        tr_list.append(
            FitResult(
                success=False,
                params=np.array([]),
                param_names=[],
                cov=None,
                residuals=None,
                chi2=None,
                message="—",
            )
        )

    for i in range(n):
        conc = titration.concentration[i]
        lab = titration.labels[i] if titration.labels and i < len(titration.labels) else ""
        res = tr_list[i]
        if (
            model_type == "piecewise"
            and res.success
            and len(res.params) >= 4
            and len(res.param_names) >= 4
        ):
            sts = _param_stderrs(res)
            row = [
                f"{conc:g}",
                str(lab) if lab else "—",
                _fmt_pm(float(res.params[0]), sts[0] if len(sts) > 0 else None),
                _fmt_pm(float(res.params[1]), sts[1] if len(sts) > 1 else None),
                _fmt_pm(float(res.params[2]), sts[2] if len(sts) > 2 else None),
                _fmt_pm(float(res.params[3]), sts[3] if len(sts) > 3 else None),
                f"{float(res.chi2):.4g}" if res.chi2 is not None else "—",
            ]
        else:
            msg = (res.message or "failed")[:32]
            row = [
                f"{conc:g}",
                str(lab) if lab else "—",
                "—",
                "—",
                "—",
                "—",
                msg,
            ]
        rows.append(row)

    if not rows:
        rows = [["—", "—", "—", "—", "—", "—", "No traces"]]

    table = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.65)

    ax_gl = fig.add_subplot(gs[1])
    ax_gl.axis("off")
    ax_gl.set_title("Global fit on titration series", fontsize=11, fontweight="bold", pad=6)

    if model_type != "piecewise":
        gtxt = f"Global summary not implemented for model_type={model_type!r}."
    elif global_result.success and len(global_result.params) >= 2:
        k_on = float(global_result.params[0])
        k_off = float(global_result.params[1])
        extra = global_result.extra or {}
        K_d_M = float(extra.get("K_d", k_off / k_on if k_on else np.nan))
        K_d_nM = K_d_M * 1e9 if np.isfinite(K_d_M) else np.nan
        gtxt = (
            "Linear model:  k_obs = k_on · [L] + k_off   ([L] in M)\n\n"
            f"  k_on   = {k_on:.6g}  M⁻¹ s⁻¹   (association rate / concentration)\n"
            f"  k_off  = {k_off:.6g}  s⁻¹       (dissociation rate)\n"
            f"  K_D    = {K_d_M:.6g}  M   = {K_d_nM:.6g}  nM   (binding affinity, equilibrium dissociation constant)\n"
            "\n"
            "(Uncertainties for global parameters are not estimated from this linear regression.)"
        )
    else:
        gtxt = (
            "Global fit unavailable.\n\n"
            f"{global_result.message or 'No global parameters.'}"
        )

    ax_gl.text(
        0.02,
        0.98,
        gtxt,
        transform=ax_gl.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="0.96", edgecolor="0.75"),
    )

    return fig


def plot_kinetic_results(
    global_result: FitResult,
    titration: Optional[TitrationSeries] = None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    extra = global_result.extra or {}
    has_k_obs = "concentrations_M" in extra and "k_obs_observed" in extra

    if not has_k_obs:
        msg = global_result.message or "No global kinetics plot"
        if not global_result.success:
            msg = f"Global fit failed.\n{msg}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, wrap=True)
        ax.set_title("k_obs vs concentration (unavailable)")
        return ax

    conc_M = np.asarray(extra["concentrations_M"])
    k_obs_obs = np.asarray(extra["k_obs_observed"])
    ax.plot(conc_M * 1e9, k_obs_obs, "o", markersize=8, label="Observed k_obs")

    if global_result.success and len(global_result.params) >= 2:
        k_on = float(global_result.params[0])
        k_off = float(global_result.params[1])
        conc_plot = np.linspace(0, float(conc_M.max()) * 1.1, 50) * 1e9
        k_obs_fit = k_on * (conc_plot * 1e-9) + k_off
        ax.plot(conc_plot, k_obs_fit, "-", linewidth=2, label="Fit")
        K_d = extra.get("K_d", k_off / k_on if k_on else np.nan)
        ax.text(
            0.05,
            0.95,
            f"k_on = {k_on:.2e} M⁻¹s⁻¹\nk_off = {k_off:.2e} s⁻¹\nK_d = {K_d:.2e} M",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
    else:
        ax.text(
            0.05,
            0.95,
            f"No line fit.\n{global_result.message}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("k_obs (s⁻¹)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
