"""
Cumulative Hitter Reports
-------------------------
Builds a one-page report for every CSUN hitter from the fall TrackMan files:
overall and platoon stat lines, a spray chart, hard-hit and whiff locations,
pitch-type splits, and batted-ball / contact / swing-decision metrics compared
with the 2026 D1 average.

Inputs (same folder as this script):
  Fall Lives/*.csv            TrackMan exports from fall lives and scrimmages
  FinalStuffPlusDataset.csv   2026 D1 pitch data, used for the league averages

Output:
  CSUN_Fall_Hitter_Reports.pdf, one page per hitter, most plate appearances first
"""

import base64
import glob
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Arc, Polygon, Rectangle
from PIL import Image


# ============================================================
# Settings
# ============================================================

TEAM = "CAL_MAT"
DATA_FILES = "Fall Lives/*.csv"
D1_FILE = "FinalStuffPlusDataset.csv"
OUTPUT_PDF = "CSUN_Fall_Hitter_Reports.pdf"

# Roster cleanup
# Player names replaced for the public version of this script.
NAME_FIXES = {"Hitter, Wrong": "Hitter, A"}      # a player tagged under the wrong name
NOT_HITTING = ["Hitter, X", "Hitter, Y"]          # players no longer hitting
SWITCH_HITTERS = ["Hitter, S"]                     # get one page per side of the plate

# Pitch-call and pitch-type groups
SWINGS = ["StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
FASTBALLS = ["Fastball", "Sinker", "Cutter"]
BREAKING = ["Curveball", "Slider", "Sweeper"]
OFFSPEED = ["ChangeUp", "Changeup", "Splitter"]

# wOBA weights
WOBA_WEIGHTS = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}

# CSUN Baseball logo, top-left corner (inches), matching the pitcher reports
LOGO_LEFT_IN, LOGO_TOP_IN, LOGO_WIDTH_IN = 0.45, 0.14, 1.25
LOGO_ASPECT = 0.4375      # height / width of the embedded logo

HIT_COLORS = {"GroundBall": "#d62728", "LineDrive": "#2ca02c",
              "FlyBall": "#1f77b4", "Popup": "#9467bd", "Bunt": "#7f7f7f"}


# ============================================================
# Load and clean the data
# ============================================================

def ended_plate_appearance(df):
    """True for the pitch that ended each plate appearance."""
    return df["KorBB"].isin(["Strikeout", "Walk"]) | df["PitchCall"].isin(["InPlay", "HitByPitch"])


files = sorted(glob.glob(DATA_FILES))
df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)

hitters_df = df[df["BatterTeam"] == TEAM].copy()
hitters_df["Batter"] = hitters_df["Batter"].str.replace(r"\s+", " ", regex=True).str.strip()
hitters_df["Batter"] = hitters_df["Batter"].replace(NAME_FIXES)
hitters_df = hitters_df[~hitters_df["Batter"].isin(NOT_HITTING)]

# Switch hitters: work out which side they hit from (opposite the pitcher's hand)
# and give each side its own name, e.g. "Smith, John – as LHH".
is_switch = hitters_df["Batter"].isin(SWITCH_HITTERS)
hitters_df.loc[is_switch, "BatterSide"] = (
    hitters_df.loc[is_switch, "PitcherThrows"].map({"Right": "Left", "Left": "Right"}))
side_label = (hitters_df.loc[is_switch, "BatterSide"]
              .map({"Left": "as LHH", "Right": "as RHH"}).fillna("side unknown"))
hitters_df.loc[is_switch, "Batter"] = hitters_df.loc[is_switch, "Batter"] + " – " + side_label

plate_appearances = hitters_df[ended_plate_appearance(hitters_df)]


# ============================================================
# Stat lines and splits
# ============================================================

def stat_line(pas):
    """AVG / OBP / SLG / OPS / wOBA / HR for a set of plate-appearance-ending pitches."""
    result = pas["PlayResult"]
    singles = (result == "Single").sum()
    doubles = (result == "Double").sum()
    triples = (result == "Triple").sum()
    hr = (result == "HomeRun").sum()
    bb = (pas["KorBB"] == "Walk").sum()
    hbp = (pas["PitchCall"] == "HitByPitch").sum()
    sac = (result == "Sacrifice").sum()

    hits = singles + doubles + triples + hr
    at_bats = len(pas) - bb - hbp - sac
    total_bases = singles + 2 * doubles + 3 * triples + 4 * hr
    obp_denominator = at_bats + bb + hbp + sac

    avg = hits / at_bats if at_bats else 0
    slg = total_bases / at_bats if at_bats else 0
    obp = (hits + bb + hbp) / obp_denominator if len(pas) else 0

    w = WOBA_WEIGHTS
    woba_numerator = (w["BB"] * bb + w["HBP"] * hbp + w["1B"] * singles
                      + w["2B"] * doubles + w["3B"] * triples + w["HR"] * hr)
    woba = woba_numerator / obp_denominator if len(pas) else 0

    return {"PA": len(pas), "AVG": round(avg, 3), "OBP": round(obp, 3),
            "SLG": round(slg, 3), "OPS": round(obp + slg, 3),
            "wOBA": round(woba, 3), "HR": hr}


def overall_table(name):
    """Overall / vs RHP / vs LHP stat lines for one hitter."""
    pas = plate_appearances[plate_appearances["Batter"] == name]
    table = pd.DataFrame({
        "Overall": stat_line(pas),
        "vs RHP": stat_line(pas[pas["PitcherThrows"] == "Right"]),
        "vs LHP": stat_line(pas[pas["PitcherThrows"] == "Left"]),
    })
    return table.T.astype({"PA": int, "HR": int})


def pitch_group_row(pitches):
    """Results against one group of pitches (e.g. breaking balls from righties)."""
    line = stat_line(pitches[ended_plate_appearance(pitches)])
    swings = pitches["PitchCall"].isin(SWINGS).sum()
    whiffs = (pitches["PitchCall"] == "StrikeSwinging").sum()
    avg_ev = pitches.loc[pitches["PitchCall"] == "InPlay", "ExitSpeed"].mean()
    return {"Pitches": len(pitches),
            "AVG": line["AVG"],
            "SLG": line["SLG"],
            "Whiff%": round(whiffs / swings * 100, 1) if swings else 0,
            "Avg EV": round(avg_ev, 1) if pd.notna(avg_ev) else "-"}


def pitch_type_table(name):
    """Fastballs (both hands), then breaking and offspeed split by pitcher hand."""
    seen = hitters_df[hitters_df["Batter"] == name]
    vs_rhp = seen[seen["PitcherThrows"] == "Right"]
    vs_lhp = seen[seen["PitcherThrows"] == "Left"]
    table = pd.DataFrame({
        "Fastballs": pitch_group_row(seen[seen["TaggedPitchType"].isin(FASTBALLS)]),
        "RHP Breaking": pitch_group_row(vs_rhp[vs_rhp["TaggedPitchType"].isin(BREAKING)]),
        "RHP Offspeed": pitch_group_row(vs_rhp[vs_rhp["TaggedPitchType"].isin(OFFSPEED)]),
        "LHP Breaking": pitch_group_row(vs_lhp[vs_lhp["TaggedPitchType"].isin(BREAKING)]),
        "LHP Offspeed": pitch_group_row(vs_lhp[vs_lhp["TaggedPitchType"].isin(OFFSPEED)]),
    })
    return table.T.astype({"Pitches": int})


# ============================================================
# Batted ball, contact quality and swing decisions
# ============================================================

def in_zone(pitches):
    return (pitches["PlateLocSide"].abs() <= 0.83) & pitches["PlateLocHeight"].between(1.5, 3.5)


def metrics(pitches):
    """Batted-ball, contact and swing-decision rates for any group of pitches
    (one hitter, or the whole D1 file)."""
    def r(x):
        return round(x, 1)

    balls_in_play = pitches[pitches["PitchCall"] == "InPlay"]

    # Batted-ball types, bunts left out
    hit_types = balls_in_play.loc[
        balls_in_play["TaggedHitType"].isin(["GroundBall", "LineDrive", "FlyBall", "Popup"]),
        "TaggedHitType"]
    type_pct = hit_types.value_counts(normalize=True) * 100

    # Spray direction, with lefties flipped so negative always means pull side
    sprayed = balls_in_play.dropna(subset=["Direction"])
    sprayed = sprayed[sprayed["BatterSide"].isin(["Right", "Left"])]
    angle = np.where(sprayed["BatterSide"] == "Right", sprayed["Direction"], -sprayed["Direction"])
    n_sprayed = len(angle)

    exit_velo = balls_in_play["ExitSpeed"].dropna()
    pas = pitches[ended_plate_appearance(pitches)]

    # Swing decisions use only pitches with a tracked location
    located = pitches.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    zone = in_zone(located)
    swung = located["PitchCall"].isin(SWINGS)

    return {
        "GB%": r(type_pct.get("GroundBall", 0)),
        "LD%": r(type_pct.get("LineDrive", 0)),
        "FB%": r(type_pct.get("FlyBall", 0)),
        "PU%": r(type_pct.get("Popup", 0)),
        "Pull%": r((angle < -15).mean() * 100) if n_sprayed else 0,
        "Mid%": r(((angle >= -15) & (angle <= 15)).mean() * 100) if n_sprayed else 0,
        "Oppo%": r((angle > 15).mean() * 100) if n_sprayed else 0,
        "Avg EV": r(exit_velo.mean()) if len(exit_velo) else 0,
        "Max EV": r(exit_velo.max()) if len(exit_velo) else 0,
        "Hard Hit%": r((exit_velo >= 90).mean() * 100) if len(exit_velo) else 0,
        "BB%": r((pas["KorBB"] == "Walk").mean() * 100) if len(pas) else 0,
        "K%": r((pas["KorBB"] == "Strikeout").mean() * 100) if len(pas) else 0,
        "Chase%": r(swung[~zone].mean() * 100) if (~zone).sum() else 0,
        "Z-Swing%": r(swung[zone].mean() * 100) if zone.sum() else 0,
    }


# 2026 D1 averages (only load the columns we need; the file is large)
d1_columns = ["Batter", "BatterSide", "PitchCall", "KorBB", "TaggedHitType",
              "ExitSpeed", "Direction", "PlateLocSide", "PlateLocHeight"]
d1 = pd.read_csv(D1_FILE, usecols=d1_columns, low_memory=False)
d1_avg = metrics(d1)

# For Max EV, a league "average max" means the average of each hitter's own max
# (hitters with at least 20 balls in play), not the single hardest ball in the file.
d1_bip = d1[d1["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
hitter_max = d1_bip.groupby("Batter")["ExitSpeed"].agg(["max", "count"])
d1_avg["Max EV"] = round(hitter_max.loc[hitter_max["count"] >= 20, "max"].mean(), 1)


def compare_table(name, columns):
    player = metrics(hitters_df[hitters_df["Batter"] == name])
    return pd.DataFrame({"Player": player, "2026 Avg": d1_avg}).T[columns]


# ============================================================
# Charts
# ============================================================

def spray_chart(ax, name):
    balls = hitters_df[(hitters_df["Batter"] == name) & (hitters_df["PitchCall"] == "InPlay")]
    balls = balls.dropna(subset=["Direction", "Distance"])

    # Field outline: home plate at (0, 0), center field straight up
    ax.plot([0, -233], [0, 233], color="black", lw=1)       # left field line
    ax.plot([0, 233], [0, 233], color="black", lw=1)        # right field line
    angles = np.radians(np.linspace(-45, 45, 100))
    fence = 330 + 70 * np.cos(np.radians(np.linspace(-90, 90, 100)))  # 330 down the lines, 400 to CF
    ax.plot(fence * np.sin(angles), fence * np.cos(angles), color="black", lw=1)
    ax.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color="gray", lw=1)  # bases
    ax.add_patch(Arc((0, 60.5), 190, 190, theta1=20, theta2=160, color="gray", lw=1))  # infield arc

    # Each ball in play: spray angle + distance -> x, y
    angle = np.radians(balls["Direction"])
    x = balls["Distance"] * np.sin(angle)
    y = balls["Distance"] * np.cos(angle)
    colors = balls["TaggedHitType"].map(HIT_COLORS).fillna("black")
    ax.scatter(x, y, c=colors, s=30, edgecolor="white", linewidth=0.5, zorder=3)

    for label, color in HIT_COLORS.items():
        ax.scatter([], [], c=color, s=30, label=label)
    ax.legend(loc="center left", fontsize=6, frameon=False)
    ax.set_xlim(-400, 260)
    ax.set_ylim(-20, 430)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Spray Chart", fontsize=10, fontweight="bold")


def zone_plot(ax, pitches, title, color):
    # Strike zone from the catcher's view: 17 in wide, 1.5 to 3.5 ft tall
    ax.add_patch(Rectangle((-0.708, 1.5), 1.417, 2.0, fill=False, color="black", lw=1.5))
    # Home plate under the zone, point toward the catcher
    plate = [(-0.708, 0.9), (0.708, 0.9), (0.708, 0.75), (0, 0.6), (-0.708, 0.75)]
    ax.add_patch(Polygon(plate, closed=True, fill=False, color="black", lw=1))
    # TrackMan's PlateLocSide is from the pitcher's side, so flip it for the catcher's view
    ax.scatter(-pitches["PlateLocSide"], pitches["PlateLocHeight"],
               c=color, s=35, alpha=0.75, edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0.4, 4.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold")


def hard_hit_plot(ax, name):
    hard = hitters_df[(hitters_df["Batter"] == name) & (hitters_df["PitchCall"] == "InPlay")
                      & (hitters_df["ExitSpeed"] >= 90)]
    zone_plot(ax, hard, f"Hard Hit (90+ mph) – {len(hard)}", "#d62728")


def whiff_plot(ax, name):
    whiffs = hitters_df[(hitters_df["Batter"] == name) & (hitters_df["PitchCall"] == "StrikeSwinging")]
    zone_plot(ax, whiffs, f"Whiffs – {len(whiffs)}", "#1f77b4")


# ============================================================
# Page layout
# ============================================================

RATE_COLUMNS = ["AVG", "OBP", "SLG", "OPS", "wOBA"]


def format_cell(column, value):
    if isinstance(value, str):
        return value
    if column in RATE_COLUMNS:
        text = f"{value:.3f}"
        return text.lstrip("0") if value < 1 else text      # .269 style
    if column in ["PA", "HR", "Pitches"] and float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def draw_table(ax, table, title):
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=14)
    rows = [[label] + [format_cell(col, v) for col, v in zip(table.columns, values)]
            for label, values in zip(table.index, table.values)]
    n_cols = len(table.columns)
    widths = [0.27] + [0.73 / n_cols] * n_cols     # wider first column for the row labels
    t = ax.table(cellText=rows, colLabels=[""] + list(table.columns),
                 colWidths=widths, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.scale(1, 1.3)
    for (row, col), cell in t.get_celld().items():
        if row == 0 or col == 0:                    # header row and row labels
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")


def hitter_page(name):
    fig = plt.figure(figsize=(8.5, 11))
    grid = fig.add_gridspec(5, 2, height_ratios=[0.8, 2.4, 0.9, 2.2, 1.6],
                            hspace=0.45, wspace=0.25, left=0.08, right=0.95, top=0.9, bottom=0.04)

    side = hitters_df.loc[hitters_df["Batter"] == name, "BatterSide"].mode()
    bats = side.iloc[0][0] if len(side) else "?"
    logo_h = LOGO_WIDTH_IN * LOGO_ASPECT
    ax_logo = fig.add_axes([LOGO_LEFT_IN / 8.5, 1 - (LOGO_TOP_IN + logo_h) / 11, LOGO_WIDTH_IN / 8.5, logo_h / 11])
    ax_logo.axis("off")
    ax_logo.imshow(np.array(Image.open(io.BytesIO(base64.b64decode(LOGO_B64))).convert("RGBA")),
                   aspect="equal", interpolation="bilinear")

    fig.suptitle(f"{name}  (Bats: {bats})", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, "CSUN Baseball – Fall 2026 Hitter Report", ha="center", fontsize=10, color="gray")

    draw_table(fig.add_subplot(grid[0, :]), overall_table(name), "Overall Stats")
    spray_chart(fig.add_subplot(grid[1, 0]), name)
    hard_hit_plot(fig.add_subplot(grid[1, 1]), name)
    draw_table(fig.add_subplot(grid[2, 0]),
               compare_table(name, ["GB%", "LD%", "FB%", "PU%", "Pull%", "Mid%", "Oppo%"]),
               "Batted Ball %")
    draw_table(fig.add_subplot(grid[2, 1]),
               compare_table(name, ["Avg EV", "Max EV", "Hard Hit%"]), "Contact Quality")
    draw_table(fig.add_subplot(grid[3, 0]), pitch_type_table(name), "Pitch Type Splits")
    whiff_plot(fig.add_subplot(grid[3, 1]), name)
    draw_table(fig.add_subplot(grid[4, 1]),
               compare_table(name, ["BB%", "K%", "Chase%", "Z-Swing%"]), "Swing Decisions")

    notes = ("Data: TrackMan, Fall 2026 lives and scrimmages.\n"
             "2026 Avg: all D1 hitters in the 2026 Stuff+ dataset.\n"
             "Hard hit = 90+ mph exit velo. Zone = 17 in wide, 1.5-3.5 ft.\n"
             "Pull/Mid/Oppo = three equal 30° slices of the field.\n"
             "Charts are drawn from the catcher's view.")
    fig.text(0.08, 0.06, notes, fontsize=7, color="gray", va="bottom", linespacing=1.6)
    return fig


# ============================================================
# Build the PDF
# ============================================================

def main():
    # Order pages by plate appearances. A switch hitter's two pages are ranked by
    # his combined total so they stay next to each other.
    pa_counts = plate_appearances["Batter"].value_counts()

    def total_pa(page_name):
        player = page_name.split(" – ")[0]
        return pa_counts[pa_counts.index.str.startswith(player)].sum()

    page_order = sorted(pa_counts.index, key=lambda name: (-total_pa(name), name))

    with PdfPages(OUTPUT_PDF) as pdf:
        for name in page_order:
            fig = hitter_page(name)
            pdf.savefig(fig)
            plt.close(fig)
            print("Done:", name)

    print(f"Saved {OUTPUT_PDF} ({len(files)} TrackMan files)")


# ============================================================
# Embedded logo (PNG, base64)
# ============================================================

LOGO_B64 = (
    "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"
    "6/8Aze9Ey4+6r2sAAAAASUVORK5CYII="
)


if __name__ == "__main__":
    main()
