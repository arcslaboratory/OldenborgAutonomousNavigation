from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import KDTree

try:
    import fastdtw
    from scipy.spatial.distance import euclidean

    _FASTDTW = True
except ImportError:
    _FASTDTW = False

# Motive CSV fixed row indices (0-based)
_ROW_TYPE = 2
_ROW_NAME = 3
_ROW_SUBHEADER = 6
_ROW_AXIS = 7
_ROW_DATA = 8

COL_INDICES = [0, 1, 6, 8]  # fallback for single-body CSVs
COL_NAMES = ["frame", "time", "x", "z"]

NOISE_STD = 5.0
DRIFT_MAX = 15.0
TIME_SCALE = 1.15
FRAME_SCALE = 1.30
SMOOTH_WIN = 15

# Dark theme palette
_BG = "#0F1923"
_PANEL = "#16232E"
_GRID = "#1E3040"
_TEXT = "#E0E8F0"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _read_header_row(path: Path, row_index: int) -> list:
    df = pd.read_csv(
        path, header=None, skiprows=row_index, nrows=1, dtype=str, encoding="latin-1"
    )
    return df.iloc[0].fillna("").tolist()


def _find_body_cols(path: Path, body_name: str) -> list:
    """Locate [frame_col, time_col, x_col, z_col] for the named rigid body.

    Reads Type, Name, Subheader, and Axis label rows explicitly so that X/Z
    are matched by their label rather than by positional offset.
    """
    types = _read_header_row(path, _ROW_TYPE)
    names = _read_header_row(path, _ROW_NAME)
    subhdrs = _read_header_row(path, _ROW_SUBHEADER)
    axes = _read_header_row(path, _ROW_AXIS)

    x_col = z_col = None
    for i, (t, n, s, a) in enumerate(zip(types, names, subhdrs, axes)):
        if (
            t.strip() == "Rigid Body"
            and n.strip() == body_name
            and s.strip() == "Position"
        ):
            label = a.strip().upper()
            if label == "X":
                x_col = i
            elif label == "Z":
                z_col = i

    if x_col is None or z_col is None:
        available = sorted(
            {
                n.strip()
                for t, n in zip(types, names)
                if t.strip() == "Rigid Body" and n.strip()
            }
        )
        raise ValueError(
            f"Rigid body '{body_name}' Position X/Z not found in {path}.\n"
            f"Available rigid bodies: {available}\n"
            "Use --robot_body / --track_body to specify the correct name."
        )
    return [0, 1, x_col, z_col]


def load_motive_csv(
    path: Path, body_name: Optional[str] = None, scale: float = 1.0
) -> pd.DataFrame:
    """Load frame/time/x/z for a named rigid body from a Motive CSV.

    body_name: rigid body label in the Name row. When None, falls back to the
               hardcoded COL_INDICES (suitable for single-body track.csv).
    scale:     multiplied onto x/z after loading (e.g. 0.001 converts mm to m).
    """
    col_indices = _find_body_cols(path, body_name) if body_name else COL_INDICES
    df = pd.read_csv(path, skiprows=_ROW_DATA, header=0, encoding="latin-1")
    df = df.iloc[:, col_indices].copy()
    df.columns = COL_NAMES
    before = len(df)
    df = df.dropna().astype(float).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  ({dropped} dropout frames removed)")
    if scale != 1.0:
        df["x"] *= scale
        df["z"] *= scale
    return df


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------


def generate_mock_robot(
    track: pd.DataFrame, out_path: Path, rng: np.random.Generator
) -> pd.DataFrame:
    t0, t1 = track["time"].iloc[0], track["time"].iloc[-1]
    n = int(len(track) * FRAME_SCALE)
    robot_time = np.linspace(t0, t1 * TIME_SCALE, n)
    sample_time = np.linspace(t0, t1, n)
    track_t = track["time"].to_numpy()
    spline_x = CubicSpline(track_t, track["x"].to_numpy())
    spline_z = CubicSpline(track_t, track["z"].to_numpy())
    xs = (
        spline_x(sample_time)
        + rng.normal(0, NOISE_STD, n)
        + np.linspace(0, DRIFT_MAX, n)
    )
    zs = spline_z(sample_time) + rng.normal(0, NOISE_STD, n)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        for i in range(_ROW_DATA):
            f.write(f"# Mock robot recording - metadata row {i + 1}\n")
        f.write("Frame,Time (Seconds),,,,,X,,Z\n")
        for frame, ts, x, z in zip(np.arange(1, n + 1), robot_time, xs, zs):
            f.write(f"{int(frame)},{ts:.6f},0,0,0,0,{x:.4f},0,{z:.4f}\n")
    return pd.DataFrame(
        {
            "frame": np.arange(1, n + 1, dtype=float),
            "time": robot_time,
            "x": xs,
            "z": zs,
        }
    )


# ---------------------------------------------------------------------------
# Deviation analysis
# ---------------------------------------------------------------------------


def compute_deviation(track: pd.DataFrame, robot: pd.DataFrame) -> np.ndarray:
    tree = KDTree(track[["x", "z"]].to_numpy())
    distances, _ = tree.query(robot[["x", "z"]].to_numpy())
    return distances


def _manual_dtw(
    track_xz: np.ndarray, robot_xz: np.ndarray, band_pct: float = 0.1
) -> tuple:
    """Diagonal-centred banded DTW fallback when fastdtw is not installed.

    The band is centred on the expected diagonal (j ≈ i * N/M) so it works
    correctly even when len(track) >> len(robot).
    """
    M, N = len(track_xz), len(robot_xz)
    band = max(1, int(band_pct * max(M, N)))
    INF = float("inf")
    mat = np.full((M + 1, N + 1), INF)
    mat[0, 0] = 0.0
    for i in range(1, M + 1):
        j_center = round((i - 1) * N / M)
        for j in range(max(1, j_center - band + 1), min(N + 1, j_center + band + 1)):
            cost = np.linalg.norm(track_xz[i - 1] - robot_xz[j - 1])
            mat[i, j] = cost + min(mat[i - 1, j], mat[i, j - 1], mat[i - 1, j - 1])
    path, i, j = [], M, N
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        _, i, j = min(
            (mat[i - 1, j], i - 1, j),
            (mat[i, j - 1], i, j - 1),
            (mat[i - 1, j - 1], i - 1, j - 1),
        )
    path.reverse()
    per_pt = np.zeros(N)
    counts = np.zeros(N)
    for ti, ri in path:
        per_pt[ri] += np.linalg.norm(track_xz[ti] - robot_xz[ri])
        counts[ri] += 1
    counts = np.maximum(counts, 1)
    return float(mat[M, N]), per_pt / counts


def dtw_errors(track: pd.DataFrame, robot: pd.DataFrame) -> tuple:
    """Return (total_dtw_cost, per_robot_point_errors)."""
    t = track[["x", "z"]].to_numpy()
    r = robot[["x", "z"]].to_numpy()
    if _FASTDTW:
        total, path = fastdtw.fastdtw(r, t, dist=euclidean)
        per_pt = np.zeros(len(r))
        counts = np.zeros(len(r))
        for ri, ti in path:
            per_pt[ri] += np.linalg.norm(r[ri] - t[ti])
            counts[ri] += 1
        counts = np.maximum(counts, 1)
        return float(total), per_pt / counts
    return _manual_dtw(t, r)


# ---------------------------------------------------------------------------
# Coordinate alignment
# ---------------------------------------------------------------------------


def align_robot_to_track(track: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    """Minimize mean KDTree deviation via Nelder-Mead over (dx, dz, rotation).
    Uses centroid offset as initial guess so convergence is fast.
    """
    t = track[["x", "z"]].to_numpy()
    r = robot[["x", "z"]].to_numpy()
    tree = KDTree(t)

    def _obj(params):
        dx, dz, deg = params
        rad = np.radians(deg)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        return tree.query((R @ r.T).T + [dx, dz])[0].mean()

    dx0, dz0 = t.mean(axis=0) - r.mean(axis=0)
    res = minimize(
        _obj,
        [dx0, dz0, 0.0],
        method="Nelder-Mead",
        options={"xatol": 0.5, "fatol": 0.01, "maxiter": 2000},
    )
    dx, dz, deg = res.x
    rad = np.radians(deg)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    r_aligned = (R @ r.T).T + [dx, dz]
    print(f"Alignment: translation ({dx:+.1f}, {dz:+.1f}) mm, rotation {deg:+.2f}°")
    aligned = robot.copy()
    aligned["x"] = r_aligned[:, 0]
    aligned["z"] = r_aligned[:, 1]
    return aligned


def _sync_start(track: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    """Trim robot so its first frame is nearest to the track's first point.

    Searches only within the first estimated lap (1/N of robot frames) to avoid
    matching a later lap's crossing of the start position.
    """
    track_start = track[["x", "z"]].iloc[0].to_numpy()
    search_n = min(len(robot), max(200, len(robot) // 5))
    robot_sub = robot[["x", "z"]].iloc[:search_n].to_numpy()
    dist, idx = KDTree(robot_sub).query(track_start)
    if dist > 200:
        print(f"  Start sync skipped (nearest robot frame is {dist:.1f} mm from track start)")
        return robot
    if idx == 0:
        return robot
    print(f"  Start sync: dropped {idx} leading frames; robot now starts {dist:.1f} mm from track start")
    return robot.iloc[idx:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(errors: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "std": float(np.std(errors)),
    }


def print_summary(
    nn_errors: np.ndarray,
    dtw_cost: Optional[float] = None,
    dtw_errors_pp: Optional[np.ndarray] = None,
) -> None:
    nn = compute_metrics(nn_errors)
    sep = "=" * 52
    print(f"\n{sep}")
    print("  TRAJECTORY EVALUATION REPORT")
    print(sep)
    print("  -- Nearest-Neighbour Spatial Matching --")
    print(f"     MAE        : {nn['mae']:.2f} mm")
    print(f"     RMSE       : {nn['rmse']:.2f} mm")
    print(f"     Max Error  : {nn['max_error']:.2f} mm")
    print(f"     Std Dev    : {nn['std']:.2f} mm")
    print(f"     % <= 10 mm : {100 * (nn_errors < 10).mean():.1f}%")
    print(f"     % <= 20 mm : {100 * (nn_errors < 20).mean():.1f}%")
    print(f"     % <= 50 mm : {100 * (nn_errors < 50).mean():.1f}%")
    if dtw_cost is not None and dtw_errors_pp is not None:
        dtw = compute_metrics(dtw_errors_pp)
        print()
        print("  -- Dynamic Time Warping (Sequence-Aware) --")
        print(f"     DTW Cost   : {dtw_cost:.2f}")
        print(f"     MAE (DTW)  : {dtw['mae']:.2f} mm")
        print(f"     RMSE (DTW) : {dtw['rmse']:.2f} mm")
        print(f"     Max Error  : {dtw['max_error']:.2f} mm")
    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_dark_axes(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(color=_GRID, linewidth=0.5)


def plot_all(
    track: pd.DataFrame,
    robot: pd.DataFrame,
    nn_errors: np.ndarray,
    out_path: Path,
    dtw_errors_pp: Optional[np.ndarray] = None,
    dark: bool = True,
) -> None:
    if dark:
        sns.set_theme(style="darkgrid")
        fig_fc = _BG
        track_c = "#4FC3F7"
        robot_c = "#FF7043"
        dtw_c = "#B39DDB"
        cmap = "plasma"
        txt_c = _TEXT
    else:
        sns.reset_defaults()
        fig_fc = "white"
        track_c = "steelblue"
        robot_c = "darkorange"
        dtw_c = "purple"
        cmap = "RdYlGn_r"
        txt_c = "black"

    fig = plt.figure(figsize=(18, 10), facecolor=fig_fc)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    if dark:
        for ax in (ax1, ax2, ax3):
            _apply_dark_axes(ax)

    fig.suptitle(
        "Robot Locomotion Accuracy Analysis",
        fontsize=14,
        fontweight="bold",
        color=txt_c,
    )

    lkw = dict(
        fontsize=7,
        facecolor=fig_fc,
        labelcolor=txt_c,
        edgecolor=_GRID if dark else "gray",
        loc="best",
    )

    # Shared axis limits for Panels 1 & 2 (cover both track and robot)
    all_x = np.concatenate([track["x"].to_numpy(), robot["x"].to_numpy()])
    all_z = np.concatenate([track["z"].to_numpy(), robot["z"].to_numpy()])
    _pad = max((all_x.max() - all_x.min()) * 0.05, 1.0)
    _xlim = (all_x.min() - _pad, all_x.max() + _pad)
    _zlim = (all_z.min() - _pad, all_z.max() + _pad)

    # -- Panel 1: path overlay -----------------------------------------------
    ax1.plot(
        track["x"], track["z"], color=track_c, lw=1.5, label="Track (ground truth)"
    )
    ax1.plot(
        robot["x"], robot["z"], color=robot_c, lw=1.0, alpha=0.85, label="Robot path"
    )
    for df, mk in [(track, "o"), (robot, "^")]:
        ax1.scatter(
            df["x"].iloc[0], df["z"].iloc[0], color="lime", s=70, marker=mk, zorder=5
        )
        ax1.scatter(
            df["x"].iloc[-1],
            df["z"].iloc[-1],
            color="yellow",
            s=70,
            marker="X",
            zorder=5,
        )
    ax1.set_xlim(_xlim)
    ax1.set_ylim(_zlim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Z (mm)")
    ax1.set_title("Path Overlay")
    ax1.legend(**lkw)

    # -- Panel 2: LineCollection deviation heatmap ---------------------------
    r_pts = robot[["x", "z"]].to_numpy()
    segs = list(np.concatenate([r_pts[:-1, None], r_pts[1:, None]], axis=1))
    seg_err = (nn_errors[:-1] + nn_errors[1:]) / 2
    norm = mcolors.Normalize(vmin=nn_errors.min(), vmax=nn_errors.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.5, alpha=0.95)
    lc.set_array(seg_err)
    ax2.add_collection(lc)
    ax2.plot(
        track["x"],
        track["z"],
        color=track_c,
        lw=1,
        alpha=0.3,
        ls="--",
        zorder=1,
        label="Track",
    )
    cb = fig.colorbar(lc, ax=ax2, pad=0.02, fraction=0.046)
    cb.set_label("Deviation (mm)", color=txt_c, fontsize=8)
    if dark:
        cb.ax.yaxis.set_tick_params(color=_TEXT, labelcolor=_TEXT, labelsize=7)
        cb.outline.set_edgecolor(_GRID)  # type: ignore[union-attr]
    ax2.set_xlim(_xlim)
    ax2.set_ylim(_zlim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.set_title("Deviation Heatmap")
    ax2.legend(**lkw)

    # -- Panel 3: deviation over real time -----------------------------------
    raw_dev = pd.Series(nn_errors)
    smooth_dev = raw_dev.rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    t_axis = robot["time"]
    ax3.fill_between(t_axis, raw_dev, alpha=0.15, color=robot_c)
    ax3.plot(t_axis, raw_dev, color=robot_c, lw=0.8, label="NN deviation (raw)")
    ax3.plot(
        t_axis,
        smooth_dev,
        color=robot_c,
        lw=1.8,
        label=f"NN deviation (rolling mean, w={SMOOTH_WIN})",
    )
    ax3.axhline(10, color="gold", lw=1.2, ls=":", label="10 mm")
    ax3.axhline(20, color="tomato", lw=1.2, ls=":", label="20 mm")
    ax3.axhline(50, color="firebrick", lw=1.2, ls=":", label="50 mm")
    ax3.set_xlim(t_axis.min(), t_axis.max())
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Deviation (mm)")
    ax3.set_title("Deviation Over Time")
    ax3.legend(**{**lkw, "loc": "upper right"})

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig_fc)
    print(f"Figure saved -> {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="Robot locomotion accuracy analysis (Motive CSV)"
    )
    parser.add_argument("--track", type=str, default="track.csv")
    parser.add_argument("--robot", type=str, default="robot.csv")
    parser.add_argument("--output", type=str, default="locomotion_analysis.png")
    parser.add_argument(
        "--track_body",
        type=str,
        default=None,
        help="Rigid body name for the track (None = single-body CSV fallback)",
    )
    parser.add_argument(
        "--robot_body",
        type=str,
        default="Robot",
        help="Rigid body name for the robot in a multi-body CSV",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Coordinate scale factor (e.g. 0.001 converts mm to m)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=60,
        help="Keep every Nth robot frame (1 = no downsampling)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_align",
        default="store_true",
        help="Skip coordinate-system alignment between takes",
    )
    parser.add_argument(
        "--no_sync_start",
        action="store_true",
        help="Skip trimming robot to match the track's starting position",
    )
    parser.add_argument(
        "--no_dtw", action="store_true", help="Skip DTW analysis (faster runs)"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use light matplotlib theme instead of dark",
    )
    args = parser.parse_args()

    track_path = Path(args.track)
    robot_path = Path(args.robot)

    track_df = load_motive_csv(track_path, body_name=args.track_body, scale=args.scale)
    track_dur = track_df["time"].iloc[-1] - track_df["time"].iloc[0]
    print(f"Track loaded : {len(track_df)} frames, {track_dur:.2f} s duration")

    if not robot_path.exists():
        if robot_path != Path("robot.csv"):
            raise FileNotFoundError(
                f"{robot_path} not found. Mock data is only auto-generated for "
                "the default 'robot.csv' path to avoid overwriting real recordings."
            )
        print(f"{robot_path} not found — generating mock robot data ...")
        rng = np.random.default_rng(args.seed)
        robot_df = generate_mock_robot(track_df, robot_path, rng)
        print(f"Mock robot CSV written -> {robot_path}")
    else:
        robot_df = load_motive_csv(
            robot_path, body_name=args.robot_body, scale=args.scale
        )

    robot_dur = robot_df["time"].iloc[-1] - robot_df["time"].iloc[0]
    print(f"Robot loaded : {len(robot_df)} frames, {robot_dur:.2f} s duration")

    if args.stride > 1:
        robot_df = robot_df.iloc[:: args.stride].reset_index(drop=True)
        print(f"Robot downsampled: {len(robot_df)} frames (stride {args.stride})")

    if not args.no_align:
        robot_df = align_robot_to_track(track_df, robot_df)
    else:
        print("Alignment skipped (--no_align)")

    if not args.no_sync_start:
        robot_df = _sync_start(track_df, robot_df)

    nn_errors = compute_deviation(track_df, robot_df)

    dtw_cost, dtw_pp = None, None
    if not args.no_dtw:
        if not _FASTDTW:
            print("[INFO] fastdtw not installed — using manual DTW fallback.")
        print("Computing DTW...")
        dtw_cost, dtw_pp = dtw_errors(track_df, robot_df)

    print_summary(nn_errors, dtw_cost, dtw_pp)

    track_plot = track_df.iloc[:: args.stride].reset_index(drop=True)
    out = Path(args.output)
    n = 1
    while True:
        candidate = out.with_name(f"{out.stem}_{n}{out.suffix}")
        if not candidate.exists():
            break
        n += 1

    plot_all(
        track_plot,
        robot_df,
        nn_errors,
        candidate,
        dtw_errors_pp=dtw_pp,
        dark=not args.light,
    )


if __name__ == "__main__":
    main()
