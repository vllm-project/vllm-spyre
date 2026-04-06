#!/usr/bin/env python3
"""
Live Spyre accelerator monitor — plots power (W) and busy (%) over time.

Usage:
    source /home/boh/dt-inductor/.venv/bin/activate
    python monitor_spyre.py [--delay SECONDS] [--window SECONDS] [--refresh-ms MS] [--port PORT]

Uses dcr_parser.py directly (bypassing the aiu-smi 1-second print gate)
to support sub-second sampling.  Default --delay is 0.5s.  Falls back
to aiu-smi if dcr_parser.py cannot be located.

Display forwarding from a pod (WebAgg)
---------------------------------------
This script uses the matplotlib WebAgg backend, which serves the live plot
as a web page.  No X11 or GUI toolkit needed on the remote machine.

  1. Connect to the pod with port forwarding:

       ssh -L 8888:localhost:8888 boh-spyre-dev-vllm-ssh

  2. On the pod, run:

       source /home/boh/dt-inductor/.venv/bin/activate
       python monitor_spyre.py              # serves on port 8888 by default
       python monitor_spyre.py --port 9999  # or pick a different port

  3. On your local notebook, open in a browser:

       http://localhost:8888

The live plots will update in the browser automatically.
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use("WebAgg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# ── IBM color palette ────────────────────────────────────────────────
IBM_DARK_GRAY = "#21272a"
IBM_MID_GRAY = "#697077"
IBM_LIGHT_GRAY = "#f2f4f8"
IBM_BLUE = "#0f62fe"
IBM_BLUE_GLOW = "#0f62fe"  # line
IBM_BLUE_FILL = "#0f62fe18"  # translucent fill under curve

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Live Spyre monitor",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--delay", type=float, default=0.5,
    help="sample interval in seconds (default 0.5)",
)
parser.add_argument(
    "--window", type=int, default=120,
    help="visible time window in seconds (default 120)",
)
parser.add_argument(
    "--refresh-ms", type=int, default=500,
    help="plot refresh interval in milliseconds (default 500)",
)
parser.add_argument(
    "--port", type=int, default=8888,
    help="WebAgg server port (default 8888)",
)
parser.add_argument(
    "--hide-time", action="store_true",
    help="hide time labels on x-axis (keep ticks only)",
)
parser.add_argument(
    "--power-arbitrary", action="store_true",
    help="show power on an arbitrary scale (hide W values and y-axis labels)",
)
parser.add_argument(
    "--no-power", action="store_true",
    help="disable the power plot, only show the busy plot",
)
args = parser.parse_args()
args.delay = max(0.1, args.delay)

# Configure WebAgg
matplotlib.rcParams["webagg.port"] = args.port
matplotlib.rcParams["webagg.open_in_browser"] = False
matplotlib.rcParams["webagg.address"] = "0.0.0.0"

# ── Patch WebAgg HTML so browser background matches the plot ─────────
import tempfile, shutil, atexit
from pathlib import Path
import matplotlib.backends.backend_webagg_core as _webagg_core
import matplotlib.backends.backend_webagg as _bw

_orig_web = Path(_webagg_core.FigureManagerWebAgg.get_static_file_path())
_custom_web = Path(tempfile.mkdtemp(prefix="spyre_monitor_"))
atexit.register(lambda: shutil.rmtree(_custom_web, ignore_errors=True))

# Symlink everything from the original web_backend directory
for item in _orig_web.iterdir():
    (_custom_web / item.name).symlink_to(item)

# Dark-theme CSS snippet shared by both pages
_DARK_CSS = f"""\
    <style>
      html, body {{
        background-color: {IBM_DARK_GRAY} !important;
        margin: 0; padding: 0;
      }}
      #figure, #figures {{
        display: flex; justify-content: center;
        margin: 0 !important; padding: 8px 0 0 0 !important;
      }}
      .mpl-toolbar {{
        background-color: {IBM_DARK_GRAY} !important;
        border-top: 1px solid {IBM_MID_GRAY} !important;
      }}
      .mpl-toolbar .mpl-widget {{ filter: invert(0.85); }}
      .mpl-message {{ color: {IBM_LIGHT_GRAY} !important; }}
    </style>"""

# Overwrite single_figure.html
(_custom_web / "single_figure.html").unlink()
(_custom_web / "single_figure.html").write_text(f"""\
<!DOCTYPE html>
<html lang="en">
  <head>
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/page.css" type="text/css">
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/boilerplate.css" type="text/css">
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/fbm.css" type="text/css">
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/mpl.css" type="text/css">
    <script src="{{{{ prefix }}}}/_static/js/mpl_tornado.js"></script>
    <script src="{{{{ prefix }}}}/js/mpl.js"></script>
{_DARK_CSS}
    <script>
      function ready(fn) {{
        if (document.readyState != "loading") {{ fn(); }}
        else {{ document.addEventListener("DOMContentLoaded", fn); }}
      }}
      ready(function () {{
        var websocket_type = mpl.get_websocket_type();
        var uri = "{{{{ ws_uri }}}}" + {{{{ str(fig_id) }}}} + "/ws";
        if (window.location.protocol === 'https:') uri = uri.replace('ws:', 'wss:');
        var websocket = new websocket_type(uri);
        var fig = new mpl.figure(
            {{{{ str(fig_id) }}}}, websocket, mpl_ondownload,
            document.getElementById("figure"));
      }});
    </script>
    <title>Spyre Monitor</title>
  </head>
  <body>
    <div id="mpl-warnings" class="mpl-warnings"></div>
    <div id="figure"></div>
  </body>
</html>
""")

# Overwrite all_figures.html (served at /)
(_custom_web / "all_figures.html").unlink()
(_custom_web / "all_figures.html").write_text(f"""\
<!DOCTYPE html>
<html lang="en">
  <head>
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/page.css" type="text/css">
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/boilerplate.css" type="text/css">
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/fbm.css" type="text/css">
    <link rel="stylesheet" href="{{{{ prefix }}}}/_static/css/mpl.css" type="text/css">
    <script src="{{{{ prefix }}}}/_static/js/mpl_tornado.js"></script>
    <script src="{{{{ prefix }}}}/js/mpl.js"></script>
{_DARK_CSS}
    <script>
      function ready(fn) {{
        if (document.readyState != "loading") {{ fn(); }}
        else {{ document.addEventListener("DOMContentLoaded", fn); }}
      }}
      function figure_ready(fig_id) {{
        return function () {{
          var main_div = document.querySelector("div#figures");
          var figure_div = document.createElement("div");
          figure_div.id = "figure-div";
          main_div.appendChild(figure_div);
          var websocket_type = mpl.get_websocket_type();
          var uri = "{{{{ ws_uri }}}}" + fig_id + "/ws";
          if (window.location.protocol === "https:") uri = uri.replace('ws:', 'wss:');
          var websocket = new websocket_type(uri);
          var fig = new mpl.figure(fig_id, websocket, mpl_ondownload, figure_div);
          fig.focus_on_mouseover = true;
          fig.canvas.setAttribute("tabindex", fig_id);
        }}
      }}
      {{% for (fig_id, fig_manager) in figures %}}
        ready(figure_ready({{{{ str(fig_id) }}}}));
      {{% end %}}
    </script>
    <title>Spyre Monitor</title>
  </head>
  <body>
    <div id="mpl-warnings" class="mpl-warnings"></div>
    <div id="figures"></div>
  </body>
</html>
""")

# Tell WebAgg to use our custom directory
_webagg_core.FigureManagerWebAgg.get_static_file_path = staticmethod(
    lambda: str(_custom_web)
)
_orig_webapp_init = _bw.WebAggApplication.__init__

def _patched_webapp_init(self_app, *a, **kw):
    _orig_webapp_init(self_app, *a, **kw)
    self_app.settings["template_path"] = str(_custom_web)

_bw.WebAggApplication.__init__ = _patched_webapp_init

# ── shared state ─────────────────────────────────────────────────────
max_pts = args.window + 120
timestamps = deque(maxlen=max_pts)
pwr_data = deque(maxlen=max_pts)
busy_data = deque(maxlen=max_pts)
lock = threading.Lock()

# ── locate dcr_parser.py ─────────────────────────────────────────────
def _find_dcr_parser():
    """Find dcr_parser.py and its lib directory next to the aiu-smi script."""
    import shutil as _sh
    aiu = _sh.which("aiu-smi")
    if aiu is None:
        return None, None
    aiu = Path(aiu).resolve()
    for candidate in [aiu.parent / "dcr_parser.py",
                      aiu.parent.parent / "lib" / "dcr_parser.py"]:
        if candidate.is_file():
            return str(candidate), str(candidate.parent)
    return None, None

_dcr_parser_path, _dcr_lib_dir = _find_dcr_parser()

# ── background reader ────────────────────────────────────────────────
def _build_reader_cmd():
    """
    Build the command and env to read Spyre metrics.

    Prefers dcr_parser.py directly with a sub-second interval and
    smi_interval patched to 0 so every sample is printed.  Falls back
    to aiu-smi if dcr_parser.py cannot be found.

    Returns (cmd_list, env_dict).
    """
    import os as _os
    if _dcr_parser_path is not None:
        # Convert delay to the format dcr_parser.py understands.
        delay_ms = int(args.delay * 1000)
        interval_arg = f"{delay_ms}ms" if delay_ms < 1000 else str(args.delay)
        # Write a small wrapper script that imports dcr_parser and
        # monkey-patches two things:
        #   1. smi_interval = 0  (print every sample, bypass 1s gate)
        #   2. Override the read-loop sleep to use our desired interval
        #      (aiu-smi mode normally forces 10ms reads with a 1s print
        #       gate; we want to read AND print at the user's --delay)
        wrapper_path = _custom_web / "_dcr_wrapper.py"
        wrapper_path.write_text(
            f"import sys\n"
            f"sys.argv = ['dcr_parser.py', '--aiu-smi', '--csv', '-i', '{interval_arg}']\n"
            f"import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('dcr_parser', r'{_dcr_parser_path}')\n"
            f"mod = importlib.util.module_from_spec(spec)\n"
            f"spec.loader.exec_module(mod)\n"
            f"\n"
            f"# Patch 1: bypass the 1-second print gate\n"
            f"_orig_parse = mod.parseFilesForAiuSmi\n"
            f"def _patched(files, args):\n"
            f"    mod.smi_interval = 0\n"
            f"    return _orig_parse(files, args)\n"
            f"mod.parseFilesForAiuSmi = _patched\n"
            f"\n"
            f"# Patch 2: override calcInterval so the aiu-smi code path\n"
            f"# keeps our desired interval instead of replacing it with 10ms\n"
            f"_real_calcInterval = mod.calcInterval\n"
            f"_call_count = 0\n"
            f"def _patched_calcInterval(s):\n"
            f"    global _call_count\n"
            f"    _call_count += 1\n"
            f"    val = _real_calcInterval(s)\n"
            f"    # The 2nd call in main() is calcInterval(smi_raw_interval)\n"
            f"    # which forces 10ms; return our interval instead.\n"
            f"    if _call_count == 2:\n"
            f"        return _real_calcInterval('{interval_arg}')\n"
            f"    return val\n"
            f"mod.calcInterval = _patched_calcInterval\n"
            f"\n"
            f"mod.main()\n"
        )
        env = _os.environ.copy()
        env["PYTHONPATH"] = _dcr_lib_dir + ":" + env.get("PYTHONPATH", "")
        return [sys.executable, "-u", str(wrapper_path)], env
    # Fallback: use aiu-smi (minimum 1 s)
    delay_s = max(1, int(args.delay))
    return ["aiu-smi", "-s", "-d", str(delay_s)], None


def reader():
    cmd, env = _build_reader_cmd()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=env,
    )
    header_map = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("ID,"):
            cols = [c.strip().split()[0] for c in line.split(",")]
            header_map = {name: idx for idx, name in enumerate(cols)}
            continue
        if line.startswith("#"):
            continue
        if header_map is None:
            continue

        parts = line.split(",")
        if len(parts) < len(header_map):
            continue

        try:
            # Use wall-clock time for sub-second precision (the CSV
            # timestamp from dcr_parser only has whole-second resolution).
            ts = datetime.now()

            pwr_raw = parts[header_map["pwr"]].strip()
            busy_raw = parts[header_map["busy"]].strip()

            pwr = float(pwr_raw) if pwr_raw not in ("-", "") else None
            busy = float(busy_raw) if busy_raw not in ("-", "") else None
        except (ValueError, KeyError):
            continue

        with lock:
            timestamps.append(ts)
            pwr_data.append(pwr)
            busy_data.append(busy)


t = threading.Thread(target=reader, daemon=True)
t.start()

# ── wait for first sample ────────────────────────────────────────────
print("Waiting for aiu-smi data ...", end="", flush=True)
for _ in range(30):
    if timestamps:
        break
    time.sleep(0.5)
print(" ok" if timestamps else " (no data yet, will start when data arrives)")

# ── figure setup ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": IBM_DARK_GRAY,
    "axes.facecolor": IBM_DARK_GRAY,
    "axes.edgecolor": IBM_MID_GRAY,
    "axes.labelcolor": IBM_LIGHT_GRAY,
    "text.color": IBM_LIGHT_GRAY,
    "xtick.color": IBM_LIGHT_GRAY,
    "ytick.color": IBM_LIGHT_GRAY,
    "grid.color": IBM_MID_GRAY,
    "grid.alpha": 0.15,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 12,
})

# ── Use fig.set_label to control the WebAgg page title ───────────────
_nrows = 1 if args.no_power else 2
_fig_h = 4.0 if args.no_power else 6.5
fig = plt.figure("Spyre Monitor", figsize=(9, _fig_h))
fig.set_label("Spyre Monitor")

gs = fig.add_gridspec(
    _nrows, 1,
    hspace=0.50,
    left=0.10, right=0.94,
    top=0.90, bottom=0.08,
)

ax_pwr = None
if not args.no_power:
    ax_pwr = fig.add_subplot(gs[0, 0])
    ax_busy = fig.add_subplot(gs[1, 0])
else:
    ax_busy = fig.add_subplot(gs[0, 0])

fig.canvas.manager.set_window_title("Spyre Monitor")

# ── dashboard title ──────────────────────────────────────────────────
fig.text(
    0.52, 0.96, "IBM Spyre Accelerator Monitor",
    ha="center", va="center",
    fontsize=17, fontweight="bold", color=IBM_LIGHT_GRAY,
    fontfamily="sans-serif",
)

# ── style helper ─────────────────────────────────────────────────────
def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.grid(True, linewidth=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(IBM_MID_GRAY)
    ax.spines["bottom"].set_color(IBM_MID_GRAY)
    if args.hide_time:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="x", rotation=0, labelsize=10, pad=4)
    ax.tick_params(axis="y", labelsize=10, pad=4)


line_pwr = fill_pwr = txt_pwr = None
if ax_pwr is not None:
    pwr_title = "Load" if args.power_arbitrary else "Power"
    pwr_ylabel = "a.u." if args.power_arbitrary else "Watts"
    style_ax(ax_pwr, pwr_title, pwr_ylabel)
    if args.power_arbitrary:
        ax_pwr.yaxis.set_major_formatter(plt.NullFormatter())
    (line_pwr,) = ax_pwr.plot([], [], color=IBM_BLUE_GLOW, linewidth=2.0)
    fill_pwr = ax_pwr.fill_between([], [], alpha=0.0)
    if not args.power_arbitrary:
        txt_pwr = ax_pwr.text(
            1.0, 1.06, "", transform=ax_pwr.transAxes,
            ha="right", va="center", fontsize=14, fontweight="bold",
            color=IBM_BLUE, fontfamily="monospace",
        )

style_ax(ax_busy, "Busy", "%")
(line_busy,) = ax_busy.plot([], [], color=IBM_BLUE_GLOW, linewidth=2.0)
fill_busy = ax_busy.fill_between([], [], alpha=0.0)
txt_busy = ax_busy.text(
    1.0, 1.06, "", transform=ax_busy.transAxes,
    ha="right", va="center", fontsize=14, fontweight="bold",
    color=IBM_BLUE, fontfamily="monospace",
)

# ── animation loop ───────────────────────────────────────────────────
def update():
    global fill_pwr, fill_busy

    with lock:
        if not timestamps:
            return
        ts = list(timestamps)
        pwr = list(pwr_data)
        busy = list(busy_data)

    ts_pwr = [t for t, v in zip(ts, pwr) if v is not None]
    val_pwr = [v for v in pwr if v is not None]
    ts_busy = [t for t, v in zip(ts, busy) if v is not None]
    val_busy = [v for v in busy if v is not None]

    if ts_pwr and line_pwr is not None:
        line_pwr.set_data(ts_pwr, val_pwr)
        fill_pwr.remove()
        fill_pwr = ax_pwr.fill_between(
            ts_pwr, val_pwr, color=IBM_BLUE, alpha=0.10,
        )
        ax_pwr.relim()
        ax_pwr.autoscale_view()
        if txt_pwr is not None:
            txt_pwr.set_text(f"{val_pwr[-1]:.0f} W")

    if ts_busy:
        line_busy.set_data(ts_busy, val_busy)
        fill_busy.remove()
        fill_busy = ax_busy.fill_between(
            ts_busy, val_busy, color=IBM_BLUE, alpha=0.10,
        )
        ax_busy.set_ylim(-2, 105)
        ax_busy.relim()
        ax_busy.autoscale_view(scaley=False)
        txt_busy.set_text(f"{val_busy[-1]:.0f} %")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


timer = fig.canvas.new_timer(interval=args.refresh_ms)
timer.add_callback(update)
timer.start()

print(f"\nServing live plots at http://localhost:{args.port}")
print("Open this URL in your local browser (requires SSH port forwarding).")
plt.show()
