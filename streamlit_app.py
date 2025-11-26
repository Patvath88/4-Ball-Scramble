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

# -------------------------- THEME (high-contrast) -----------------------------
GOLF_CSS = """
<style>
:root { --golf-green:#0b7a24; --golf-dark:#064a15; --sand:#f3e6c1; }
section.main { background: radial-gradient(1200px 600px at 10% -10%, #ffffff, #eaf7e7) }
.block-container { padding-top: 0.8rem; }
.golf-hero{padding:.8rem 1rem;border-radius:12px;background:linear-gradient(135deg,var(--golf-green),var(--golf-dark));color:#fff;display:flex;align-items:center;gap:14px}
.golf-badge{background:#ffffff22;padding:6px 10px;border-radius:10px;font-weight:800}
.team-card{border:3px solid var(--golf-green);border-radius:16px;padding:14px;background:#fff;box-shadow:0 2px 10px rgba(0,0,0,.08)}
.player-chip{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:999px;border:2px solid var(--golf-green);background:#fff;opacity:1 !important;margin:6px 6px}
.player-name{font-weight:900;font-size:1.05rem;color:#0c0c0c !important}
.rank-pill{background:var(--golf-green);color:#fff;border-radius:999px;font-weight:900;min-width:36px;height:36px;padding:0 8px;display:inline-flex;align-items:center;justify-content:center}
.leaderboard{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.lb-card{border-radius:14px;padding:12px 14px;background:#fff;border:2px solid #e7e7e7;box-shadow:0 2px 10px rgba(0,0,0,.06)}
.lb-card .title{font-weight:900;font-size:1.05rem}
.lb-card .meta{display:flex;gap:10px;align-items:center;margin-top:6px}
.lb-1{border-color:#f1c40f;background:linear-gradient(180deg,#fffdf0,#fff8c7)}
.lb-2{border-color:#bdc3c7;background:linear-gradient(180deg,#ffffff,#f2f4f6)}
.lb-3{border-color:#cd7f32;background:linear-gradient(180deg,#fff7f0,#fde0c8)}
.lb-rank{font-weight:900;border-radius:999px;padding:6px 10px}
.lb-1 .lb-rank{background:#f1c40f}
.lb-2 .lb-rank{background:#bdc3c7}
.lb-3 .lb-rank{background:#cd7f32}
.small{color:#333;font-size:.92rem}
/* Primary buttons -> red; used for End Round only */
div.stButton > button[kind="primary"]{ background:#c0392b;border-color:#8e2a1b;color:#fff; }
div.stButton > button[kind="primary"]:hover{ background:#a33224; }
@media (prefers-color-scheme: dark){
  .team-card,.player-chip,.lb-card{background:#fff}
  .player-name{color:#0d0d0d !important}
}
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
    hole_winners: List[Optional[str]]              # "Team A" | "Team B" | None
    history: List[Dict]                            # [{'hole':n,'Team A':[],'Team B':[],'Winner':str}]
    show_results: bool
    started_at: float

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
    split = max(1, len(sh)//2)
    return {"Team A": sh[:split], "Team B": sh[split:]}

def set_players(players: List[str]) -> None:
    rs: RoundState = st.session_state.rs
    rs.players = players
    rs.points = {p: rs.points.get(p, 0) for p in players}
    if not any(rs.teams.values()):
        rs.teams = random_pair(players)

def record_winner(team_name: str) -> None:
    rs: RoundState = st.session_state.rs
    idx = rs.current_hole - 1
    if idx < 0 or idx > 17 or rs.hole_winners[idx] is not None:
        return
    # award
    for p in rs.teams.get(team_name, []):
        rs.points[p] = rs.points.get(p, 0) + 1
    # log
    rs.hole_winners[idx] = team_name
    rs.history.append({
        "hole": rs.current_hole,
        "Team A": rs.teams["Team A"][:],
        "Team B": rs.teams["Team B"][:],
        "Winner": team_name,
    })
    # next
    if rs.current_hole == 18:
        rs.show_results = True
    else:
        rs.current_hole += 1
        rs.teams = random_pair(rs.players)

def adjust_point(player: str, delta: int) -> None:
    rs: RoundState = st.session_state.rs
    rs.points[player] = max(0, rs.points.get(player, 0) + delta)

def set_point(player: str, value: int) -> None:
    rs: RoundState = st.session_state.rs
    rs.points[player] = max(0, int(value))

def results_df() -> pd.DataFrame:
    rs: RoundState = st.session_state.rs
    df = pd.DataFrame([{"Player": p, "Points": rs.points.get(p, 0)} for p in rs.players])
    return df.sort_values(by=["Points", "Player"], ascending=[False, True]).reset_index(drop=True)

def dense_rank(points: Dict[str, int]) -> Dict[str, int]:
    """Dense ranking: 1,1,2,3 for ties."""
    sorted_players = sorted(points.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks: Dict[str, int] = {}
    last_pts = None
    rank = 0
    for i, (p, pts) in enumerate(sorted_players):
        if pts != last_pts:
            rank += 1
            last_pts = pts
        ranks[p] = rank
    return ranks

def current_streak(player: str) -> int:
    """Consecutive recent holes where player's team won."""
    rs: RoundState = st.session_state.rs
    streak = 0
    for entry in reversed(rs.history):
        winners = entry["Team A"] if entry["Winner"] == "Team A" else entry["Team B"]
        if player in winners:
            streak += 1
        else:
            break
    return streak

def combo_stats() -> pd.DataFrame:
    rs: RoundState = st.session_state.rs
    if len(rs.players) < 2:
        return pd.DataFrame(columns=["Pair", "Times Teamed", "% of 18"])
    counts: Dict[Tuple[str, str], int] = {tuple(sorted(pair)): 0 for pair in combinations(rs.players, 2)}
    for entry in rs.history:
        for team in ("Team A", "Team B"):
            for a, b in combinations(sorted(entry[team]), 2):
                counts[(a, b)] += 1
    rows = [{"Pair": f"{a} + {b}", "Times Teamed": n, "% of 18": round(100 * n / 18, 1)}
            for (a, b), n in counts.items()]
    return pd.DataFrame(rows).sort_values(by=["Times Teamed", "Pair"], ascending=[False, True])

def make_podium_image(df: pd.DataFrame) -> bytes:
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (8, 100, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, H - 180), (W, H)], fill=(14, 122, 36))
    slots = [
        {"x": W // 2 - 120, "w": 240, "h": 210, "color": (255, 215, 0), "rank": 1},
        {"x": W // 2 - 320, "w": 220, "h": 150, "color": (192, 192, 192), "rank": 2},
        {"x": W // 2 + 120, "w": 220, "h": 110, "color": (205, 127, 50), "rank": 3},
    ]
    draw.rounded_rectangle([(30, 20), (W - 30, 90)], 16, fill=(255, 255, 255))
    draw.text((50, 35), "Round Results", fill=(10, 80, 20), font=_font(36))
    base_y = H - 180
    for s in slots:
        x0 = s["x"]; y0 = base_y - s["h"]
        draw.rounded_rectangle([(x0, y0), (x0 + s["w"], base_y)], 12, fill=s["color"])
        draw.text((x0 + 10, y0 - 30), f"#{s['rank']}", fill=(255, 255, 255), font=_font(26))
    for i, s in enumerate(slots):
        if i >= len(df): continue
        row = df.iloc[i]; text = f"{row['Player']}  •  {int(row['Points'])} pts"
        tw = draw.textlength(text, font=_font(24))
        draw.text((s["x"] + (s["w"] - tw) / 2, base_y - s["h"] + 20), text, fill=(0, 0, 0), font=_font(24))
    draw.text((W - 240, H - 26), "Made with Streamlit ⛳", fill=(255, 255, 255), font=_font(18))
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

# ------------------------------ UI COMPONENTS ---------------------------------
def chip_with_editor(player: str, points: int, rank: int) -> None:
    """Show rank (not points) on the chip; controls to edit points."""
    col_chip, col_plus, col_minus, col_num = st.columns([3, 1, 1, 1.3])
    with col_chip:
        st.markdown(
            f"<div class='player-chip'>🏁 <span class='player-name'>{player}</span> "
            f"<span class='rank-pill'>{rank}</span></div>",
            unsafe_allow_html=True,
        )
    with col_plus:
        if st.button("➕", key=f"inc_{player}", use_container_width=True):
            adjust_point(player, +1); st.rerun()
    with col_minus:
        if st.button("➖", key=f"dec_{player}", use_container_width=True):
            adjust_point(player, -1); st.rerun()
    with col_num:
        new_val = st.number_input(f"{player} pts", min_value=0, max_value=99,
                                  value=int(points), key=f"num_{player}", label_visibility="collapsed")
        if new_val != points:
            set_point(player, new_val)

def team_block_editable(team_name: str, players: List[str], points: Dict[str, int], ranks: Dict[str, int]) -> None:
    st.markdown(f"#### {team_name}")
    with st.container(border=True):
        for p in players:
            chip_with_editor(p, points.get(p, 0), ranks.get(p, 0))

def render_leaderboard(points: Dict[str, int], players: List[str]) -> None:
    """Cards ordered by points with rank + points + streak."""
    ranks = dense_rank(points)
    order = sorted(players, key=lambda p: (-points.get(p, 0), p))
    st.subheader("Leaderboard")
    # Fancy cards
    st.markdown("<div class='leaderboard'>", unsafe_allow_html=True)
    for idx, p in enumerate(order, start=1):
        cls = "lb-1" if idx == 1 else "lb-2" if idx == 2 else "lb-3" if idx == 3 else "lb-card"
        base_cls = f"lb-card {cls}"
        pts = points.get(p, 0)
        streak = current_streak(p)
        html = f"""
<div class="{base_cls}">
  <div class="title">{p}</div>
  <div class="meta">
    <span class="lb-rank">#{ranks[p]}</span>
    <span>Points: <b>{pts}</b></span>
    <span>Streak: <b>{streak}</b></span>
  </div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------ APP (one page) --------------------------------
def main():
    st.set_page_config(page_title="Golf Round – One Page", page_icon="⛳", layout="wide")
    init_state()
    st.markdown(GOLF_CSS, unsafe_allow_html=True)

    rs: RoundState = st.session_state.rs

    # Results block (auto after 18)
    if rs.show_results:
        st.success("Round complete! 🎉 Final results below.")
        df_res = results_df()
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        top = df_res.head(3)
        st.markdown(
            f"""
<div class="podium">
  <div class="slot second"><div style="font-weight:800;">2nd</div><div style="font-weight:900;">{(top.iloc[1]['Player'] if len(top)>1 else '—')}</div><div class="small">Points: {(int(top.iloc[1]['Points']) if len(top)>1 else 0)}</div></div>
  <div class="slot first"><div style="font-weight:800;">1st</div><div style="font-weight:900;">{(top.iloc[0]['Player'] if len(top)>0 else '—')}</div><div class="small">Points: {(int(top.iloc[0]['Points']) if len(top)>0 else 0)}</div></div>
  <div class="slot third"><div style="font-weight:800;">3rd</div><div style="font-weight:900;">{(top.iloc[2]['Player'] if len(top)>2 else '—')}</div><div class="small">Points: {(int(top.iloc[2]['Points']) if len(top)>2 else 0)}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        png = make_podium_image(df_res)
        st.download_button("🖼️ Save Results Poster (PNG)", data=png, file_name="golf_results.png", mime="image/png")
        st.download_button("📄 Save Standings (CSV)", data=df_res.to_csv(index=False).encode("utf-8"),
                           file_name="golf_standings.csv", mime="text/csv")
        if rs.history:
            st.download_button("📄 Save Hole Log (CSV)",
                               data=pd.DataFrame(rs.history).to_csv(index=False).encode("utf-8"),
                               file_name="golf_hole_log.csv", mime="text/csv")
            st.download_button("📄 Save Combo Stats (CSV)",
                               data=combo_stats().to_csv(index=False).encode("utf-8"),
                               file_name="golf_combo_stats.csv", mime="text/csv")

    # Header
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}'
                f'<div><div class="golf-badge">Golf Round – Teams & Score</div>'
                f'<div class="small">Randomize every hole • Names lock after Hole 1 • End Round resets</div></div></div>',
                unsafe_allow_html=True)

    # Names disappear after first hole winner is recorded
    names_locked = (rs.hole_winners[0] is not None)
    with st.container(border=True):
        if not names_locked:
            c1, c2, c3, c4 = st.columns(4)
            inputs = [
                c1.text_input("Player 1", value=(rs.players[0] if len(rs.players) > 0 else "")),
                c2.text_input("Player 2", value=(rs.players[1] if len(rs.players) > 1 else "")),
                c3.text_input("Player 3 (optional)", value=(rs.players[2] if len(rs.players) > 2 else "")),
                c4.text_input("Player 4 (optional)", value=(rs.players[3] if len(rs.players) > 3 else "")),
            ]
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button("✅ Set / Update Players", use_container_width=True):
                    try:
                        players = sanitize_players(inputs)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        set_players(players)
                        if not rs.history:
                            rs.teams = random_pair(players)
                        st.success("Players updated.")
            with b2:
                if st.button("🎲 Randomize Teams now", use_container_width=True, disabled=not rs.players):
                    rs.teams = random_pair(rs.players); st.rerun()
        else:
            st.markdown("**Players are locked for this round (after Hole 1). Use _End Round_ to change.**")
            roster = " • ".join(rs.players) if rs.players else "—"
            st.markdown(f"Current players: **{roster}**")

    if not rs.players:
        st.info("Enter 2 or 4 names above to begin.")
    else:
        # Compute ranks once and pass to chips
        ranks = dense_rank(rs.points)

        # Teams + inline point editors (chips show rank)
        colA, colB = st.columns(2)
        with colA: team_block_editable("Team A", rs.teams["Team A"], rs.points, ranks)
        with colB: team_block_editable("Team B", rs.teams["Team B"], rs.points, ranks)

        st.divider()
        st.subheader(f"Hole {rs.current_hole} / 18 • Record Winner (auto-randomizes next)")
        disabled = rs.hole_winners[rs.current_hole-1] is not None if 1 <= rs.current_hole <= 18 else True
        wA, wB, m = st.columns([1, 1, 2])
        with wA:
            if st.button("🏆 Team A won", use_container_width=True, disabled=disabled or rs.show_results):
                record_winner("Team A"); st.rerun()
        with wB:
            if st.button("🏆 Team B won", use_container_width=True, disabled=disabled or rs.show_results):
                record_winner("Team B"); st.rerun()
        with m:
            st.metric("Holes recorded", sum(1 for w in rs.hole_winners if w is not None))

        # Leaderboard (ordered by points) with streaks
        render_leaderboard(rs.points, rs.players)

        # Hole log
        st.subheader("Hole Log")
        if rs.history:
            log_df = pd.DataFrame(rs.history)[["hole", "Team A", "Team B", "Winner"]].rename(columns={"hole": "Hole"})
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No holes recorded yet.")

        # Teammate combo stats
        st.subheader("Teammate Combo Stats (out of 18)")
        st.dataframe(combo_stats(), use_container_width=True, hide_index=True)

    # --- End Round (red) ------------------------------------------------------
    st.markdown("---")
    if st.button("🛑 End Round", type="primary", use_container_width=True, help="Reset everything and start fresh"):
        st.session_state.pop("rs", None)
        init_state()
        st.success("Round reset. Enter player names to begin.")
        st.rerun()

if __name__ == "__main__":
    main()
