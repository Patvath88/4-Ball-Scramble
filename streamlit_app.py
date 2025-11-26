# file: streamlit_app.py
import io
import random
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# -------------------------- THEME & FX CSS (Masters board) --------------------
GOLF_CSS = """
<style>
:root {
  --masters-green:#0b5d1e;
  --masters-deep:#084717;
  --trim-gold:#d4af37;
  --card-ink:#0b0e11;
  --chip-green:#0b7a24;
}
section.main{background:radial-gradient(1200px 620px at 15% -20%, #ffffff, #eaf7e7)}
.block-container{padding-top:.8rem}

/* Header */
.golf-hero{padding:.8rem 1rem;border-radius:12px;background:linear-gradient(135deg,var(--chip-green),#064a15);color:#fff;display:flex;align-items:center;gap:14px}
.golf-badge{background:#ffffff22;padding:6px 10px;border-radius:10px;font-weight:800}

/* Team chips */
.player-chip{position:relative;display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:14px;border:2px solid var(--chip-green);background:#fff;margin:6px 6px;flex-wrap:wrap}
.player-name{font-weight:900;font-size:1.05rem;color:#0c0c0c}
.player-meta{font-weight:800;color:var(--chip-green);background:#eaf7e7;border:1px solid #bde0c2;border-radius:999px;padding:4px 10px}
.tie-badge{font-weight:900;background:#fff3bf;color:#8a6700;border:1px solid #e3c200;border-radius:8px;padding:2px 8px}

/* +1 animation */
.plus1{position:absolute;right:-8px;top:-10px;background:#10b981;color:#fff;border-radius:999px;
       padding:2px 8px;font-weight:900;border:2px solid #0f9e70;opacity:0;transform:translateY(6px) scale(.8);
       animation:plusOne 1s ease-out forwards}
.lb-plus1{margin-left:8px;background:#10b981;color:#fff;border:1px solid #0f9e70;border-radius:999px;padding:2px 8px;font-weight:900;opacity:0;transform:translateY(6px) scale(.8);animation:plusOne 1s ease-out forwards}
@keyframes plusOne{0%{opacity:0;transform:translateY(6px) scale(.8)}
 15%{opacity:1;transform:translateY(-2px) scale(1)} 80%{opacity:.9}
 100%{opacity:0;transform:translateY(-18px) scale(.9)}}

/* ------- Masters-style Leaderboard ------- */
.masters-wrap{margin-top:.3rem;border:3px solid var(--trim-gold);border-radius:18px;
  background:
    linear-gradient(0deg, #0e6b22, #0e6b22) padding-box,
    linear-gradient(135deg, #ffe793, var(--trim-gold)) border-box;
  padding:10px; box-shadow:0 14px 40px rgba(0,0,0,.25);
}
.mast-head{display:flex;align-items:center;justify-content:space-between;
  background:linear-gradient(#0d5a1f,#0a4d1b); color:#fff;border:2px solid #0a4519;
  border-radius:12px; padding:8px 12px; margin-bottom:10px}
.mast-title{font-family:Georgia, 'Times New Roman', serif; font-weight:900; letter-spacing:.6px; font-size:1.3rem}
.mast-pill{background:#0f7226;border:1px solid #0a4d1b;border-radius:999px;padding:4px 10px;font-weight:800}

/* board rows */
.mrow{display:grid;grid-template-columns:74px 1fr auto; align-items:center;
  background:#0f1420; border:1px solid #1f2630; border-radius:12px; margin:8px 4px; color:#eef2f7;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04), 0 6px 18px rgba(0,0,0,.25)}
.mcell{padding:10px 12px}
.rank-medal{display:flex;align-items:center;justify-content:center; gap:6px; font-weight:900; border-radius:10px;
  border:2px solid #2a3546; background:#0a0f1a; padding:10px 12px; width:66px}
.rank-medal.gold{background:linear-gradient(#3e2e00,#2c2000); border-color:#a08100; color:#ffd44d}
.rank-medal.silver{background:linear-gradient(#3a3f47,#2b2f36); border-color:#b9c5d1; color:#e3edf7}
.rank-medal.bronze{background:linear-gradient(#3a2e23,#2a2119); border-color:#d19662; color:#f1c197}
.player-box{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.pname{font-family:Georgia,'Times New Roman',serif;font-weight:900;font-size:1.2rem;letter-spacing:.3px;color:#fff}
.tie-note{background:#fff3bf;color:#8a6700;border:1px solid #e3c200;border-radius:8px;padding:2px 8px;font-weight:900}
.badges{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-right:10px}
.badge{background:#0a0f1a;border:1px solid #2a3546;border-radius:999px;padding:6px 12px;font-weight:800;color:#dfe7f4}

/* Confirm (blocking pane) */
.confirm-wrap{max-width:560px;margin:16vh auto 0 auto;}
.confirm-pane{background:#111827;border:2px solid #374151;border-radius:16px;padding:18px 18px 14px 18px;
  box-shadow:0 18px 60px rgba(0,0,0,.5); color:#fff}
.confirm-title{font-weight:900;font-size:1.25rem;margin-bottom:.4rem}
.confirm-text{color:#e5e7eb;margin-bottom:.8rem}
</style>
"""

GOLF_SVG = """
<svg width="48" height="48" viewBox="0 0 64 64" fill="none">
  <circle cx="26" cy="22" r="12" fill="#ffffff" stroke="#e6e6e6" stroke-width="2"/>
  <circle cx="22" cy="18" r="2" fill="#dcdcdc"/><circle cx="30" cy="20" r="2" fill="#dcdcdc"/>
  <circle cx="24" cy="24" r="2" fill="#dcdcdc"/><circle cx="28" cy="16" r="2" fill="#dcdcdc"/>
  <rect x="42" y="6" width="2" height="30" fill="#7a5230"/><path d="M44 6 L60 12 L44 18 Z" fill="#c0392b"/>
  <rect x="14" y="40" width="36" height="8" rx="4" fill="#0b7a24"/>
</svg>
"""

# ------------------------------ STATE -----------------------------------------
@dataclass
class RoundState:
    players: List[str]
    teams: Dict[str, List[str]]
    points: Dict[str, int]
    current_hole: int
    hole_winners: List[Optional[str]]
    history: List[Dict]
    show_results: bool
    started_at: float
    fx_armed: bool
    fx_tick: int
    show_end_confirm: bool
    toast: Optional[str]
    plus1_players: List[str]
    plus1_until: float

def init_state() -> None:
    if "rs" not in st.session_state:
        st.session_state.rs = RoundState(
            players=[],
            teams={"Team A": [], "Team B": []},
            points={},
            current_hole=1,
            hole_winners=[None] * 18,
            history=[],
            show_results=False,
            started_at=time.time(),
            fx_armed=False,
            fx_tick=0,
            show_end_confirm=False,
            toast=None,
            plus1_players=[],
            plus1_until=0.0,
        )
    upgrade_state()

def upgrade_state() -> None:
    rs = st.session_state.rs
    if not hasattr(rs, "history"): rs.history = []
    if not hasattr(rs, "show_results"): rs.show_results = False
    if not hasattr(rs, "hole_winners") or rs.hole_winners is None: rs.hole_winners = [None]*18
    if len(rs.hole_winners) != 18: rs.hole_winners = (rs.hole_winners + [None]*18)[:18]
    if not hasattr(rs, "current_hole") or rs.current_hole < 1: rs.current_hole = 1
    if not hasattr(rs, "teams") or not isinstance(rs.teams, dict): rs.teams = {"Team A": [], "Team B": []}
    if not hasattr(rs, "points") or not isinstance(rs.points, dict): rs.points = {}
    if not hasattr(rs, "players"): rs.players = []
    if not hasattr(rs, "started_at"): rs.started_at = time.time()
    if not hasattr(rs, "fx_armed"): rs.fx_armed = False
    if not hasattr(rs, "fx_tick"): rs.fx_tick = 0
    if not hasattr(rs, "show_end_confirm"): rs.show_end_confirm = False
    if not hasattr(rs, "toast"): rs.toast = None
    if not hasattr(rs, "plus1_players"): rs.plus1_players = []
    if not hasattr(rs, "plus1_until"): rs.plus1_until = 0.0

# ------------------------------ HELPERS ---------------------------------------
def sanitize_players(inputs: List[str]) -> List[str]:
    seen = set(); out = []
    for raw in inputs:
        name = raw.strip()
        if name and name.lower() not in seen:
            out.append(name); seen.add(name.lower())
    if len(out) not in (2, 4):
        raise ValueError("Enter exactly 2 or 4 distinct names.")
    return out

def random_pair(players: List[str]) -> Dict[str, List[str]]:
    sh = players[:]; random.shuffle(sh)
    split = len(sh)//2 if len(sh) >= 2 else 1
    return {"Team A": sh[:split], "Team B": sh[split:]}

def set_players(players: List[str]) -> None:
    rs: RoundState = st.session_state.rs
    rs.players = players
    rs.points = {p: rs.points.get(p, 0) for p in players}
    if not any(rs.teams.values()):
        rs.teams = random_pair(players)

def ordinal(n: int) -> str:
    return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def compute_ties(players: List[str], points: Dict[str, int]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    ordered = sorted(((p, points.get(p, 0)) for p in players), key=lambda kv: (-kv[1], kv[0]))
    labels: Dict[str, str] = {}; notes: Dict[str, str] = {}; ranks: Dict[str, int] = {}
    i = 0; rank = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        group = [p for p,_ in ordered[i:j]]
        rank += 1
        if len(group) > 1:
            for p in group:
                labels[p] = f"T-{rank}"; ranks[p] = rank
                others = [x for x in group if x != p]
                notes[p] = f"tied with {', '.join(others)} for {ordinal(rank)}"
        else:
            labels[group[0]] = str(rank); ranks[group[0]] = rank; notes[group[0]] = ""
        i = j
    return labels, notes, ranks

def record_winner(team_name: str) -> None:
    rs: RoundState = st.session_state.rs
    idx = rs.current_hole - 1
    if idx < 0 or idx > 17 or rs.hole_winners[idx] is not None:
        return
    winners = [p for p in rs.teams.get(team_name, []) if p in rs.players]
    for p in winners:
        rs.points[p] = rs.points.get(p, 0) + 1
    rs.hole_winners[idx] = team_name
    rs.history.append({"hole": rs.current_hole, "Team A": rs.teams["Team A"][:], "Team B": rs.teams["Team B"][:], "Winner": team_name})
    rs.toast = f"🏆 Hole {rs.current_hole}: {team_name} – +1 to " + ", ".join(winners) if winners else f"🏆 Hole {rs.current_hole}: {team_name}"
    rs.plus1_players = winners; rs.plus1_until = time.time() + 1.2
    if rs.current_hole == 18:
        rs.show_results = True
    else:
        rs.current_hole += 1; rs.teams = random_pair(rs.players)

def undo_last_hole() -> None:
    rs: RoundState = st.session_state.rs
    if not rs.history: return
    last = rs.history.pop()
    hole = last["hole"]; winner_team = last["Winner"]
    losers = []
    for p in last[winner_team]:
        if p in rs.players:
            rs.points[p] = max(0, rs.points.get(p, 0) - 1); losers.append(p)
    rs.hole_winners[hole - 1] = None; rs.current_hole = hole
    rs.teams = {"Team A": last["Team A"], "Team B": last["Team B"]}
    rs.show_results = False; rs.toast = f"↩️ Undid hole {hole}: removed 1 from " + ", ".join(losers) if losers else f"↩️ Undid hole {hole}"
    rs.plus1_players = []

def results_df() -> pd.DataFrame:
    rs: RoundState = st.session_state.rs
    df = pd.DataFrame([{"Player": p, "Points": rs.points.get(p, 0)} for p in rs.players])
    return df.sort_values(by=["Points", "Player"], ascending=[False, True]).reset_index(drop=True)

# ------------------------------ FX (animation only) ---------------------------
def render_fx():
    rs: RoundState = st.session_state.rs
    if not rs.fx_armed: return
    rs.fx_armed = False; rs.fx_tick += 1
    st.markdown(f"""
<div class="fx-area" id="fx{rs.fx_tick}">
  <div class="fx-ball"></div>
  <div class="fx-shadow"></div>
</div>""", unsafe_allow_html=True)

# ------------------------------ IMAGE (poster) --------------------------------
def make_podium_image(df: pd.DataFrame) -> bytes:
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (8, 100, 40)); draw = ImageDraw.Draw(img)
    draw.rectangle([(0, H - 180), (W, H)], fill=(14, 122, 36))
    slots = [
        {"x": W // 2 - 120, "w": 240, "h": 210, "color": (255, 215, 0), "rank": 1},
        {"x": W // 2 - 320, "w": 220, "h": 150, "color": (192, 192, 192), "rank": 2},
        {"x": W // 2 + 120, "w": 220, "h": 110, "color": (205, 127, 50), "rank": 3},
    ]
    draw.rounded_rectangle([(30, 20), (W - 30, 90)], 16, fill=(255, 255, 255))
    draw.text((50, 35), "Round Results", fill=(10, 80, 20), font=_font(36))
    base_y = H - 180
    for i, s in enumerate(slots):
        x0 = s["x"]; y0 = base_y - s["h"]
        draw.rounded_rectangle([(x0, y0), (x0 + s["w"], base_y)], 12, fill=s["color"])
        draw.text((x0 + 10, y0 - 30), f"#{s['rank']}", fill=(255, 255, 255), font=_font(26))
        if i < len(df):
            row = df.iloc[i]
            draw.text((x0 + 16, y0 + 18), f"{row['Player']}  •  {int(row['Points'])} pts", fill=(0, 0, 0), font=_font(24))
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def _font(size: int):
    try: return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception: return ImageFont.load_default()

# ------------------------------ UI PARTS --------------------------------------
def chip_static(player: str, points: int, place_label: str, tie_note: str, show_plus1: bool) -> None:
    tied_html = f"<span class='tie-badge'>({tie_note})</span>" if tie_note else ""
    plus1_html = "<span class='plus1'>+1</span>" if show_plus1 else ""
    st.markdown(
        f"<div class='player-chip'>🏁 <span class='player-name'>{player}</span> {tied_html} "
        f"<span class='player-meta'>Current Place: {place_label}</span> "
        f"<span class='player-meta'>Total Points: {points}</span>{plus1_html}</div>",
        unsafe_allow_html=True,
    )

def team_block_static(team_name: str, players: List[str], points: Dict[str, int],
                      labels: Dict[str, str], notes: Dict[str, str], active_plus1: set) -> None:
    st.markdown(f"#### {team_name}")
    with st.container(border=True):
        for p in players:
            chip_static(p, points.get(p, 0), labels.get
