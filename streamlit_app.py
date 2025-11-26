# file: streamlit_app.py
import io
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# --- THEME & BRANDING ---------------------------------------------------------
GOLF_BG_CSS = """
<style>
:root { --golf-green:#0b7a24; --golf-light:#eaf7e7; --golf-dark:#064a15; --sand:#f3e6c1; }
section.main { background: radial-gradient(1200px 600px at 10% -10%, #ffffff, var(--golf-light)) }
.block-container { padding-top: 1.2rem; }
.golf-hero {
  padding: 0.8rem 1rem; border-radius: 12px;
  background: linear-gradient(135deg, var(--golf-green), var(--golf-dark));
  color: #fff; display:flex; align-items:center; gap:14px;
}
.golf-badge { background:#ffffff22; padding:6px 10px; border-radius:10px; font-weight:600; }
.team-card {
  border: 2px solid var(--golf-green); border-radius: 12px; padding: 12px; background:#fff;
}
.podium { display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 10px; align-items:end; }
.podium .slot { text-align:center; padding:10px; border-radius:10px; background: var(--sand); }
.podium .first { height: 220px; background: linear-gradient(180deg,#ffd700,#f1c40f); }
.podium .second { height: 160px; background: linear-gradient(180deg,#c0c0c0,#bdc3c7); }
.podium .third { height: 120px; background: linear-gradient(180deg,#cd7f32,#b16b28); }
.player-chip { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
  border:1px solid #ddd; background:#fff; }
.score-pill { background: var(--golf-green); color:#fff; padding:2px 8px; border-radius:999px; font-weight:700; }
.small { color:#333; font-size:0.92rem; }
hr { border:0; border-top:1px solid #e7e7e7; margin:0.7rem 0; }
.confirm-danger { background:#fff5f5; border:1px solid #ffd6d6; padding:10px; border-radius:10px; }
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

# --- STATE --------------------------------------------------------------------
@dataclass
class RoundState:
    players: List[str]
    teams: Dict[str, List[str]]
    points: Dict[str, int]
    current_hole: int
    hole_winners: List[Optional[str]]  # "Team A" | "Team B" | None
    strokes: Dict[str, List[Optional[int]]]
    par: List[int]
    round_active: bool
    teams_locked: bool
    started_at: float

def init_state() -> None:
    """Initialize session state once; keeps data for the browser session."""
    if "rs" in st.session_state:
        return
    st.session_state.rs = RoundState(
        players=[],
        teams={"Team A": [], "Team B": []},
        points={},
        current_hole=1,
        hole_winners=[None] * 18,
        strokes={},
        par=[4,4,4,3,4,5,4,3,4,4,5,4,3,4,4,5,3,4],  # generic mix; editable
        round_active=False,
        teams_locked=False,
        started_at=time.time(),
    )

# --- HELPERS ------------------------------------------------------------------
def sanitize_players(inputs: List[str]) -> List[str]:
    """Trim, dedupe while preserving order; enforce 2 or 4 players."""
    seen = set()
    out = []
    for raw in inputs:
        name = raw.strip()
        if name and name.lower() not in seen:
            out.append(name)
            seen.add(name.lower())
    if len(out) not in (2, 4):
        raise ValueError("Enter exactly 2 or 4 distinct names.")
    return out

def random_pair(players: List[str]) -> Dict[str, List[str]]:
    """Randomly create two teams (2v2 for 4 players, 1v1 for 2 players)."""
    shuffled = players[:]
    random.shuffle(shuffled)
    if len(shuffled) == 2:
        return {"Team A": [shuffled[0]], "Team B": [shuffled[1]]}
    return {"Team A": shuffled[:2], "Team B": shuffled[2:]}

def start_round(players: List[str], teams: Dict[str, List[str]]) -> None:
    """Reset per-round data."""
    rs: RoundState = st.session_state.rs
    rs.players = players
    rs.teams = teams
    rs.points = {p: 0 for p in players}
    rs.current_hole = 1
    rs.hole_winners = [None] * 18
    rs.strokes = {p: [None] * 18 for p in players}
    rs.round_active = True
    rs.teams_locked = False
    rs.started_at = time.time()

def award_hole(team_name: str) -> None:
    """Allocate a point to each member of the winning team and advance hole."""
    rs: RoundState = st.session_state.rs
    idx = rs.current_hole - 1
    if idx < 0 or idx > 17:
        return
    if rs.hole_winners[idx] is not None:
        return  # already recorded
    winners = rs.teams.get(team_name, [])
    for p in winners:
        rs.points[p] = rs.points.get(p, 0) + 1
    rs.hole_winners[idx] = team_name
    rs.teams_locked = True  # prevent re-rolling during round
    if rs.current_hole < 18:
        rs.current_hole += 1

def undo_last_hole() -> None:
    """Revert the most recent hole entry."""
    rs: RoundState = st.session_state.rs
    # Find last recorded hole
    last_idx = None
    for i in range(17, -1, -1):
        if rs.hole_winners[i] is not None:
            last_idx = i
            break
    if last_idx is None:
        return
    team_name = rs.hole_winners[last_idx]
    for p in rs.teams.get(team_name or "", []):
        rs.points[p] = max(0, rs.points.get(p, 0) - 1)
    rs.hole_winners[last_idx] = None
    rs.current_hole = last_idx + 1

def compute_score_vs_par(par: List[int], strokes: List[Optional[int]]) -> Tuple[int, int, int]:
    """Return (strokes_total_filled, par_total_for_filled, diff)."""
    s_total = 0
    p_total = 0
    for p, s in zip(par, strokes):
        if s is not None:
            s_total += int(s)
            p_total += int(p)
    return s_total, p_total, s_total - p_total

def build_results_table() -> pd.DataFrame:
    """Assemble standings: points, strokes total, +/- to par."""
    rs: RoundState = st.session_state.rs
    rows = []
    for p in rs.players:
        s_total, p_total, diff = compute_score_vs_par(rs.par, rs.strokes[p])
        rows.append({"Player": p, "Points": rs.points.get(p, 0), "Strokes": s_total, "+/-": diff})
    df = pd.DataFrame(rows).sort_values(
        by=["Points", "+/-", "Strokes", "Player"], ascending=[False, True, True, True]
    ).reset_index(drop=True)
    df.index = df.index + 1
    return df

def make_podium_image(df: pd.DataFrame) -> bytes:
    """Render a simple podium PNG for top 3 using Pillow."""
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (8, 100, 40))
    draw = ImageDraw.Draw(img)
    # Soft field
    draw.rectangle([(0, H - 180), (W, H)], fill=(14, 122, 36))
    # Podium positions
    slots = [
        {"x": W // 2 - 120, "w": 240, "h": 210, "color": (255, 215, 0), "rank": 1},
        {"x": W // 2 - 320, "w": 220, "h": 150, "color": (192, 192, 192), "rank": 2},
        {"x": W // 2 + 120, "w": 220, "h": 110, "color": (205, 127, 50), "rank": 3},
    ]
    # Title
    title = "Round Results"
    draw.rounded_rectangle([(30, 20), (W - 30, 90)], 16, fill=(255, 255, 255))
    draw.text((50, 35), title, fill=(10, 80, 20), font=_font(36))
    # Draw podium blocks
    base_y = H - 180
    for s in slots:
        x0 = s["x"]
        y0 = base_y - s["h"]
        draw.rounded_rectangle([(x0, y0), (x0 + s["w"], base_y)], 12, fill=s["color"])
        draw.text((x0 + 10, y0 - 30), f"#{s['rank']}", fill=(255, 255, 255), font=_font(26))
    # Write names + points
    for i, s in enumerate(slots):
        if i >= len(df):
            continue
        row = df.iloc[i]
        name = str(row["Player"])
        points = int(row["Points"])
        text = f"{name}  •  {points} pts"
        tw, th = draw.textlength(text, font=_font(24)), 26
        tx = s["x"] + (s["w"] - tw) / 2
        ty = base_y - s["h"] + 20
        draw.text((tx, ty), text, fill=(0, 0, 0), font=_font(24))
    # Footer
    draw.text((W - 240, H - 26), "Made with Streamlit ⛳", fill=(255, 255, 255), font=_font(18))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _font(size: int) -> ImageFont.FreeTypeFont:
    """Fallback to default PIL bitmap font if truetype not found."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

# --- PAGES --------------------------------------------------------------------
def page_team_maker():
    st.markdown(GOLF_BG_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}<div><div class="golf-badge">Team Maker</div>'
                f'<div class="small">Enter 2 or 4 names • Randomize teams • Record hole winners</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        name_inputs = [
            c1.text_input("Player 1", value=(rs.players[0] if len(rs.players) > 0 else "")),
            c2.text_input("Player 2", value=(rs.players[1] if len(rs.players) > 1 else "")),
            c3.text_input("Player 3 (optional)", value=(rs.players[2] if len(rs.players) > 2 else "")),
            c4.text_input("Player 4 (optional)", value=(rs.players[3] if len(rs.players) > 3 else "")),
        ]
        btn_cols = st.columns([1,1,1,2])
        with btn_cols[0]:
            if st.button("🎲 Randomize Teams", use_container_width=True, disabled=rs.teams_locked):
                try:
                    players = sanitize_players(name_inputs)
                except ValueError as e:
                    st.error(str(e))
                else:
                    teams = random_pair(players)
                    start_round(players, teams)
                    st.success("Teams locked in. Good luck!")
        with btn_cols[1]:
            if st.button("🔁 Re-roll", use_container_width=True, disabled=rs.teams_locked):
                try:
                    players = sanitize_players(name_inputs)
                except ValueError as e:
                    st.error(str(e))
                else:
                    start_round(players, random_pair(players))

        with btn_cols[2]:
            if st.button("🔄 Reset Names", use_container_width=True, disabled=rs.round_active and rs.teams_locked):
                st.session_state.rs.players = []
                st.rerun()

    if not rs.players:
        st.info("Add 2 or 4 player names, then click **Randomize Teams**.")
        return

    # Teams & Hole Controls
    ta, tb = st.columns(2)
    with ta:
        st.markdown("#### Team A")
        st.markdown(team_block(rs.teams["Team A"], rs.points), unsafe_allow_html=True)
    with tb:
        st.markdown("#### Team B")
        st.markdown(team_block(rs.teams["Team B"], rs.points), unsafe_allow_html=True)

    st.divider()
    st.subheader(f"Hole {rs.current_hole if rs.current_hole<=18 else 18} • Winner")
    win_choice = st.radio(
        "Pick winner for the current hole",
        options=["Team A", "Team B"],
        horizontal=True,
        label_visibility="collapsed",
        disabled=rs.current_hole > 18 or not rs.round_active,
    )
    cL, cR, cU = st.columns([1,1,1])
    with cL:
        if st.button("✅ Record Hole", use_container_width=True, disabled=rs.current_hole > 18 or not rs.round_active):
            award_hole(win_choice)
    with cR:
        if st.button("↩️ Undo Last Hole", use_container_width=True):
            undo_last_hole()
    with cU:
        st.metric("Current Hole", value=min(rs.current_hole, 18), delta=f"Played: {sum(1 for w in rs.hole_winners if w)}")

    st.markdown("##### Round Safety")
    with st.container(border=True):
        st.write("Delete all round data (cannot be undone).")
        danger = st.checkbox("I understand and want to delete this round", value=False)
        if st.button("🗑️ Delete round data", type="primary", disabled=not danger):
            init_state()  # ensure state exists
            st.session_state.pop("rs", None)
            init_state()
            st.success("Round data deleted.")
            st.rerun()

def team_block(players: List[str], points: Dict[str, int]) -> str:
    """HTML for team roster with score pills."""
    if not players:
        return "<div class='team-card'>No players</div>"
    chips = "".join(
        f"<span class='player-chip'>⛳ {p} <span class='score-pill'>{points.get(p,0)}</span></span>"
        for p in players
    )
    return f"<div class='team-card' style='display:flex;gap:8px;flex-wrap:wrap'>{chips}</div>"

def page_team_points():
    st.markdown(GOLF_BG_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}<div><div class="golf-badge">Team Points</div>'
                f'<div class="small">Live points per player & per team</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs
    if not rs.players:
        st.info("Go to **Team Maker** to start a round.")
        return
    # Summary metrics
    tA_pts = sum(rs.points.get(p, 0) for p in rs.teams["Team A"])
    tB_pts = sum(rs.points.get(p, 0) for p in rs.teams["Team B"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Team A points", tA_pts)
    m2.metric("Team B points", tB_pts)
    m3.metric("Holes recorded", sum(1 for w in rs.hole_winners if w is not None))

    # Per-player table
    rows = [{"Player": p, "Points": rs.points.get(p, 0)} for p in rs.players]
    df = pd.DataFrame(rows).sort_values(by=["Points", "Player"], ascending=[False, True])
    st.dataframe(df, use_container_width=True, hide_index=True)

def page_round_scores():
    st.markdown(GOLF_BG_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}<div><div class="golf-badge">Round Scores</div>'
                f'<div class="small">Enter strokes for 18 holes and set Par</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs
    if not rs.players:
        st.info("Go to **Team Maker** to set players first.")
        return

    # Par editor
    with st.container(border=True):
        st.write("Par per hole")
        par_cols = st.columns(9)
        new_par = []
        for i in range(18):
            with par_cols[i % 9]:
                val = st.number_input(f"P{i+1}", min_value=3, max_value=6, value=int(rs.par[i]), key=f"par_{i}")
                new_par.append(int(val))
        rs.par = new_par

    st.write(" ")
    st.write("Per-player strokes (leave blank if hole not played yet)")
    # Build strokes editor
    holes = [f"H{i}" for i in range(1, 19)]
    data = []
    for p in rs.players:
        row = {"Player": p}
        for i in range(18):
            row[holes[i]] = rs.strokes[p][i]
        data.append(row)
    edit_df = pd.DataFrame(data)
    edited = st.data_editor(
        edit_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={h: st.column_config.NumberColumn(h, min_value=1, max_value=15) for h in holes},
    )
    # Sync back
    for _, r in edited.iterrows():
        pname = r["Player"]
        rs.strokes[pname] = [int(r[h]) if pd.notna(r[h]) else None for h in holes]

    # Totals
    out_rows = []
    for p in rs.players:
        s_total, p_total, diff = compute_score_vs_par(rs.par, rs.strokes[p])
        out_rows.append({"Player": p, "Strokes Total": s_total, "Par for Holes Played": p_total, "+/-": diff})
    summary = pd.DataFrame(out_rows).sort_values(by="Player")
    st.dataframe(summary, use_container_width=True, hide_index=True)

def page_results():
    st.markdown(GOLF_BG_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}<div><div class="golf-badge">Results</div>'
                f'<div class="small">Final standings, podium, and downloads</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs
    if not rs.players:
        st.info("Go to **Team Maker** to set players first.")
        return

    df = build_results_table()
    st.subheader("Standings")
    st.dataframe(df, use_container_width=True)

    # CSS Podium
    st.subheader("Podium")
    top = df.head(3).copy()
    # Render podium grid
    c = st.container()
    with c:
        st.markdown(
            """
<div class="podium">
  <div class="slot second">
    <div style="font-weight:700;">2nd</div>
    <div>{p2}</div>
    <div class="small">Points: {pts2}</div>
  </div>
  <div class="slot first">
    <div style="font-weight:700;">1st</div>
    <div>{p1}</div>
    <div class="small">Points: {pts1}</div>
  </div>
  <div class="slot third">
    <div style="font-weight:700;">3rd</div>
    <div>{p3}</div>
    <div class="small">Points: {pts3}</div>
  </div>
</div>
""".format(
                p1=(top.iloc[0]["Player"] if len(top) > 0 else "—"),
                pts1=(int(top.iloc[0]["Points"]) if len(top) > 0 else 0),
                p2=(top.iloc[1]["Player"] if len(top) > 1 else "—"),
                pts2=(int(top.iloc[1]["Points"]) if len(top) > 1 else 0),
                p3=(top.iloc[2]["Player"] if len(top) > 2 else "—"),
                pts3=(int(top.iloc[2]["Points"]) if len(top) > 2 else 0),
            ),
            unsafe_allow_html=True,
        )

    # Downloads
    png_bytes = make_podium_image(df)
    st.download_button("🖼️ Download Results Poster (PNG)", data=png_bytes, file_name="golf_results.png", mime="image/png")
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button("📄 Download Standings (CSV)", data=csv_bytes, file_name="golf_standings.csv", mime="text/csv")

# --- APP ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Golf Round Assistant", page_icon="⛳", layout="wide")
    init_state()
    st.sidebar.title("⛳ Golf Round")
    page = st.sidebar.radio("Navigate", ["Team Maker", "Team Points", "Round Scores", "Results"], index=0)
    st.sidebar.caption(f"Session age: {int((time.time()-st.session_state.rs.started_at)/60)} min")
    if page == "Team Maker":
        page_team_maker()
    elif page == "Team Points":
        page_team_points()
    elif page == "Round Scores":
        page_round_scores()
    else:
        page_results()

if __name__ == "__main__":
    main()
