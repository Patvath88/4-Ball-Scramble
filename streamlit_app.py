# file: streamlit_app.py
import io
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# --- THEME / CSS --------------------------------------------------------------
GOLF_CSS = """
<style>
:root { --golf-green:#0b7a24; --golf-light:#eaf7e7; --golf-dark:#064a15; --sand:#f3e6c1; }
section.main { background: radial-gradient(1200px 600px at 10% -10%, #ffffff, var(--golf-light)) }
.block-container { padding-top: 1.1rem; }

/* Header */
.golf-hero {
  padding: .8rem 1rem; border-radius: 12px;
  background: linear-gradient(135deg, var(--golf-green), var(--golf-dark));
  color:#fff; display:flex; align-items:center; gap:14px;
}
.golf-badge { background:#ffffff22; padding:6px 10px; border-radius:10px; font-weight:800; }

/* Team cards + player chips (high contrast) */
.team-card {
  border: 3px solid var(--golf-green); border-radius: 16px; padding: 14px; background:#fff;
  box-shadow: 0 2px 10px rgba(0,0,0,.08);
}
.player-chip {
  display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border-radius:999px;
  border:2px solid var(--golf-green); background:#fff; opacity:1 !important;
  box-shadow: 0 1px 0 rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
}
.player-name { font-weight:900; font-size:1.05rem; color:#0c0c0c !important; opacity:1 !important; }
.score-pill {
  background: var(--golf-green); color:#fff; border-radius:999px; font-weight:900;
  min-width: 38px; height: 38px; padding:0 8px; display:inline-flex; align-items:center; justify-content:center;
}

/* Dark mode safety: keep names readable */
@media (prefers-color-scheme: dark){
  .team-card, .player-chip { background:#ffffff; }
  .player-name { color:#0d0d0d !important; }
}

/* Podium */
.podium { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:10px; align-items:end; }
.podium .slot { text-align:center; padding:10px; border-radius:10px; background: var(--sand); }
.podium .first { height:220px; background: linear-gradient(180deg,#ffd700,#f1c40f); }
.podium .second { height:160px; background: linear-gradient(180deg,#c0c0c0,#bdc3c7); }
.podium .third { height:120px; background: linear-gradient(180deg,#cd7f32,#b16b28); }

.small { color:#333; font-size:.92rem; }
hr { border:0; border-top:1px solid #e7e7e7; margin:.7rem 0; }
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
    hole_winners: List[Optional[str]]          # "Team A" | "Team B" | None
    strokes: Dict[str, List[Optional[int]]]    # per player, 18 slots
    par: List[int]                             # 18 ints
    round_active: bool
    started_at: float

def init_state() -> None:
    if "rs" in st.session_state:
        return
    st.session_state.rs = RoundState(
        players=[],
        teams={"Team A": [], "Team B": []},
        points={},
        current_hole=1,
        hole_winners=[None]*18,
        strokes={},
        par=[4,4,4,3,4,5,4,3,4, 4,5,4,3,4,4,5,3,4],
        round_active=False,
        started_at=time.time(),
    )

# --- HELPERS ------------------------------------------------------------------
def sanitize_players(inputs: List[str]) -> List[str]:
    seen = set(); out = []
    for raw in inputs:
        name = raw.strip()
        if name and name.lower() not in seen:
            out.append(name); seen.add(name.lower())
    if len(out) not in (2,4):
        raise ValueError("Enter exactly 2 or 4 distinct names.")
    return out

def random_pair(players: List[str]) -> Dict[str, List[str]]:
    sh = players[:]; random.shuffle(sh)
    return {"Team A": sh[:max(1,len(sh)//2)], "Team B": sh[max(1,len(sh)//2):]}

def start_round(players: List[str], teams: Dict[str, List[str]]) -> None:
    rs: RoundState = st.session_state.rs
    rs.players = players
    rs.teams = teams
    rs.points = {p: rs.points.get(p, 0) for p in players}  # keep existing if re-seeding
    for p in players:
        rs.points.setdefault(p, 0)
        rs.strokes.setdefault(p, [None]*18)
    rs.round_active = True
    rs.current_hole = max(1, rs.current_hole)

def award_hole(team_name: str) -> None:
    rs: RoundState = st.session_state.rs
    idx = rs.current_hole - 1
    if idx < 0 or idx > 17 or rs.hole_winners[idx] is not None:
        return
    for p in rs.teams.get(team_name, []):
        rs.points[p] = rs.points.get(p, 0) + 1
    rs.hole_winners[idx] = team_name
    if rs.current_hole < 18:
        rs.current_hole += 1

def undo_last_hole() -> None:
    rs: RoundState = st.session_state.rs
    last = next((i for i in range(17,-1,-1) if rs.hole_winners[i] is not None), None)
    if last is None:
        return
    team_name = rs.hole_winners[last]
    for p in rs.teams.get(team_name or "", []):
        rs.points[p] = max(0, rs.points.get(p,0)-1)
    rs.hole_winners[last] = None
    rs.current_hole = last + 1

def compute_score_vs_par(par: List[int], strokes: List[Optional[int]]) -> Tuple[int,int,int]:
    s_total = 0; p_total = 0
    for p, s in zip(par, strokes):
        if s is not None:
            s_total += int(s); p_total += int(p)
    return s_total, p_total, s_total - p_total

def build_results_table() -> pd.DataFrame:
    rs: RoundState = st.session_state.rs
    rows = []
    for p in rs.players:
        s_total, p_total, diff = compute_score_vs_par(rs.par, rs.strokes[p])
        rows.append({"Player": p, "Points": rs.points.get(p,0), "Strokes": s_total, "+/-": diff})
    df = pd.DataFrame(rows).sort_values(
        by=["Points", "+/-", "Strokes", "Player"], ascending=[False, True, True, True]
    ).reset_index(drop=True)
    df.index = df.index + 1
    return df

def make_podium_image(df: pd.DataFrame) -> bytes:
    W,H = 900,520
    img = Image.new("RGB",(W,H),(8,100,40))
    d = ImageDraw.Draw(img)
    d.rectangle([(0,H-180),(W,H)], fill=(14,122,36))
    slots = [
        {"x": W//2-120, "w":240, "h":210, "color":(255,215,0), "rank":1},
        {"x": W//2-320, "w":220, "h":150, "color":(192,192,192), "rank":2},
        {"x": W//2+120, "w":220, "h":110, "color":(205,127,50), "rank":3},
    ]
    d.rounded_rectangle([(30,20),(W-30,90)], 16, fill=(255,255,255))
    d.text((50,35), "Round Results", fill=(10,80,20), font=_font(36))
    base = H-180
    for s in slots:
        x0=s["x"]; y0=base-s["h"]
        d.rounded_rectangle([(x0,y0),(x0+s["w"],base)], 12, fill=s["color"])
        d.text((x0+10,y0-30), f"#{s['rank']}", fill=(255,255,255), font=_font(26))
    for i, s in enumerate(slots):
        if i >= len(df): continue
        row = df.iloc[i]; name = str(row["Player"]); pts = int(row["Points"])
        text = f"{name} • {pts} pts"
        tw = d.textlength(text, font=_font(24))
        d.text((s["x"] + (s["w"]-tw)/2, base - s["h"] + 20), text, fill=(0,0,0), font=_font(24))
    d.text((W-240,H-26), "Made with Streamlit ⛳", fill=(255,255,255), font=_font(18))
    buf=io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def _font(size:int):
    try: return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception: return ImageFont.load_default()

# --- UI HELPERS ---------------------------------------------------------------
def team_block(players: List[str], points: Dict[str,int]) -> str:
    if not players: return "<div class='team-card'>No players</div>"
    chips = "".join(
        f"<span class='player-chip'>🏁 <span class='player-name'>{p}</span> "
        f"<span class='score-pill'>{points.get(p,0)}</span></span>"
        for p in players
    )
    return f"<div class='team-card' style='display:flex;gap:10px;flex-wrap:wrap'>{chips}</div>"

# --- PAGES --------------------------------------------------------------------
def page_hole_and_teams():
    st.markdown(GOLF_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}'
                f'<div><div class="golf-badge">Hole & Teams</div>'
                f'<div class="small">Randomize teams every hole • Record winners • Edit scorecard</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs

    # Player inputs
    with st.container(border=True):
        c1,c2,c3,c4 = st.columns(4)
        inputs = [
            c1.text_input("Player 1", value=(rs.players[0] if len(rs.players)>0 else "")),
            c2.text_input("Player 2", value=(rs.players[1] if len(rs.players)>1 else "")),
            c3.text_input("Player 3 (optional)", value=(rs.players[2] if len(rs.players)>2 else "")),
            c4.text_input("Player 4 (optional)", value=(rs.players[3] if len(rs.players)>3 else "")),
        ]
        b1,b2,b3 = st.columns([1,1,2])
        with b1:
            if st.button("🎲 Randomize Teams (this hole)", use_container_width=True):
                try:
                    players = sanitize_players(inputs)
                except ValueError as e:
                    st.error(str(e))
                else:
                    start_round(players, random_pair(players))
                    st.success("Teams set for this hole.")
        with b2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.pop("rs", None); init_state(); st.rerun()

    if not rs.players:
        st.info("Enter 2 or 4 names then **Randomize Teams (this hole)**.")
        return

    # Teams + points (points are shown next to names everywhere)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Team A")
        st.markdown(team_block(rs.teams["Team A"], rs.points), unsafe_allow_html=True)
    with right:
        st.markdown("#### Team B")
        st.markdown(team_block(rs.teams["Team B"], rs.points), unsafe_allow_html=True)

    st.divider()
    st.subheader(f"Record Winner • Hole {min(rs.current_hole,18)}")
    choice = st.radio("Winner", options=["Team A","Team B"], horizontal=True, label_visibility="collapsed")
    cL,cR,cM = st.columns([1,1,1])
    with cL:
        if st.button("✅ Record Hole", use_container_width=True, disabled=rs.hole_winners[rs.current_hole-1] is not None):
            award_hole(choice)
    with cR:
        if st.button("↩️ Undo Last Hole", use_container_width=True):
            undo_last_hole()
    with cM:
        st.metric("Holes recorded", sum(1 for w in rs.hole_winners if w is not None))

    # --- SCORECARD (on home page) --------------------------------------------
    st.divider()
    st.subheader("Scorecard (strokes per hole)")
    with st.container(border=True):
        st.write("Par per hole")
        cols = st.columns(9); new_par=[]
        for i in range(18):
            with cols[i%9]:
                v = st.number_input(f"P{i+1}", min_value=3, max_value=6, value=int(rs.par[i]), key=f"par_{i}")
                new_par.append(int(v))
        rs.par = new_par

    st.write("Enter strokes for each player and hole:")
    holes = [f"H{i}" for i in range(1,19)]
    data = []
    for p in rs.players:
        row = {"Player": p}
        for i in range(18): row[holes[i]] = rs.strokes[p][i]
        data.append(row)
    edit_df = pd.DataFrame(data)
    edited = st.data_editor(
        edit_df, num_rows="fixed", use_container_width=True, hide_index=True,
        column_config={h: st.column_config.NumberColumn(h, min_value=1, max_value=15) for h in holes},
    )
    # Sync strokes back
    for _, r in edited.iterrows():
        pname = r["Player"]
        st.session_state.rs.strokes[pname] = [int(r[h]) if pd.notna(r[h]) else None for h in holes]

    # Per-player totals
    rows=[]
    for p in rs.players:
        s_total, p_total, diff = compute_score_vs_par(rs.par, rs.strokes[p])
        rows.append({"Player": p, "Points": rs.points.get(p,0), "Strokes Total": s_total, "Par Played": p_total, "+/-": diff})
    st.dataframe(pd.DataFrame(rows).sort_values(by=["Points","Player"], ascending=[False,True]),
                 use_container_width=True, hide_index=True)

def page_team_points():
    st.markdown(GOLF_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}'
                f'<div><div class="golf-badge">Team Points</div>'
                f'<div class="small">Live player points (follow the player)</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs
    if not rs.players:
        st.info("Set players on **Hole & Teams** first."); return
    tA = sum(rs.points.get(p,0) for p in rs.teams["Team A"])
    tB = sum(rs.points.get(p,0) for p in rs.teams["Team B"])
    a,b,c = st.columns(3); a.metric("Team A points", tA); b.metric("Team B points", tB); c.metric("Holes recorded", sum(1 for w in rs.hole_winners if w))
    df = pd.DataFrame([{"Player":p,"Points":rs.points.get(p,0)} for p in rs.players]).sort_values(by=["Points","Player"], ascending=[False,True])
    st.dataframe(df, use_container_width=True, hide_index=True)

def page_results():
    st.markdown(GOLF_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}'
                f'<div><div class="golf-badge">Results</div>'
                f'<div class="small">Final standings, podium, and downloads</div></div></div>',
                unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs
    if not rs.players:
        st.info("Set players on **Hole & Teams** first."); return
    df = build_results_table()
    st.subheader("Standings")
    st.dataframe(df, use_container_width=True)
    st.subheader("Podium")
    top = df.head(3)
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
    png = make_podium_image(df)
    st.download_button("🖼️ Download Results Poster (PNG)", data=png, file_name="golf_results.png", mime="image/png")
    st.download_button("📄 Download Standings (CSV)", data=df.to_csv(index=True).encode("utf-8"),
                       file_name="golf_standings.csv", mime="text/csv")

# --- APP ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Golf Round Assistant", page_icon="⛳", layout="wide")
    init_state()
    st.sidebar.title("⛳ Golf Round")
    page = st.sidebar.radio("Navigate", ["Hole & Teams", "Team Points", "Results"], index=0)
    st.sidebar.caption(f"Session age: {int((time.time()-st.session_state.rs.started_at)/60)} min")
    if page == "Hole & Teams":
        page_hole_and_teams()
    elif page == "Team Points":
        page_team_points()
    else:
        page_results()

if __name__ == "__main__":
    main()
