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

# ======================= THEME & RESPONSIVE CSS ===============================
GOLF_CSS = """
<style>
:root{
  --masters-green:#0b5d1e; --masters-deep:#084717; --trim-gold:#d4af37;
  --chip-green:#0b7a24; --ink:#0b0e11;
}
html,body { overscroll-behavior-y: none; }
.block-container{ padding: .6rem .8rem; }

/* Sticky winner controls for mobile */
.sticky-ctrl{ position: sticky; top: 0; z-index: 20; padding:.4rem .5rem;
  background: linear-gradient(180deg, rgba(255,255,255,.95), rgba(255,255,255,.85));
  border: 1px solid #e7efe7; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,.08); }

/* Big finger-friendly buttons */
div.stButton > button{ height: 48px; font-weight: 900; border-radius: 12px; }
div.stButton > button[kind="primary"]{ background:#c0392b; border-color:#8e2a1b; color:#fff; }
div.stButton > button[kind="primary"]:hover{ background:#a33224; }

/* Header */
.golf-hero{padding:.8rem 1rem;border-radius:12px;background:linear-gradient(135deg,var(--chip-green),#064a15);color:#fff;display:flex;align-items:center;gap:14px}
.golf-badge{background:#ffffff22;padding:6px 10px;border-radius:10px;font-weight:800}
.player-chip{position:relative;display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:14px;border:2px solid var(--chip-green);background:#fff;margin:6px 6px;flex-wrap:wrap}
.player-name{font-weight:900;font-size:1.05rem;color:#0c0c0c}
.player-meta{font-weight:800;color:var(--chip-green);background:#eaf7e7;border:1px solid #bde0c2;border-radius:999px;padding:4px 10px}
.tie-badge,.tie-note{font-weight:900;background:#fff3bf;color:#8a6700;border:1px solid #e3c200;border-radius:8px;padding:2px 8px}
.plus1{position:absolute;right:-8px;top:-10px;background:#10b981;color:#fff;border-radius:999px;padding:2px 8px;font-weight:900;border:2px solid #0f9e70;opacity:0;transform:translateY(6px) scale(.8);animation:plusOne 1s ease-out forwards}
.lb-plus1{margin-left:8px;background:#10b981;color:#fff;border:1px solid #0f9e70;border-radius:999px;padding:2px 8px;font-weight:900;opacity:0;transform:translateY(6px) scale(.8);animation:plusOne 1s ease-out forwards}
@keyframes plusOne{0%{opacity:0;transform:translateY(6px) scale(.8)} 15%{opacity:1;transform:translateY(-2px) scale(1)} 80%{opacity:.9} 100%{opacity:0;transform:translateY(-18px) scale(.9)}}

/* Masters-style leaderboard (already implemented styles kept lightweight here) */
.masters-wrap{margin-top:.3rem;border:3px solid var(--trim-gold);border-radius:18px;
  background:linear-gradient(0deg,#0e6b22,#0e6b22) padding-box,
             linear-gradient(135deg,#ffe793,var(--trim-gold)) border-box;
  padding:10px; box-shadow:0 14px 40px rgba(0,0,0,.25);}
.mast-head{display:flex;align-items:center;justify-content:space-between;background:linear-gradient(#0d5a1f,#0a4d1b);color:#fff;border:2px solid #0a4519;border-radius:12px;padding:8px 12px;margin-bottom:10px}
.mast-title{font-family:Georgia,'Times New Roman',serif;font-weight:900;letter-spacing:.6px;font-size:1.3rem}
.mast-pill{background:#0f7226;border:1px solid #0a4d1b;border-radius:999px;padding:4px 10px;font-weight:800}
.mrow{display:grid;grid-template-columns:74px 1fr auto;align-items:center;background:#0f1420;border:1px solid #1f2630;border-radius:12px;margin:8px 4px;color:#eef2f7;box-shadow:inset 0 1px 0 rgba(255,255,255,.04), 0 6px 18px rgba(0,0,0,.25)}
.mcell{padding:10px 12px}
.rank-medal{display:flex;align-items:center;justify-content:center;gap:6px;font-weight:900;border-radius:10px;border:2px solid #2a3546;background:#0a0f1a;padding:10px 12px;width:66px}
.rank-medal.gold{background:linear-gradient(#3e2e00,#2c2000);border-color:#a08100;color:#ffd44d}
.rank-medal.silver{background:linear-gradient(#3a3f47,#2b2f36);border-color:#b9c5d1;color:#e3edf7}
.rank-medal.bronze{background:linear-gradient(#3a2e23,#2a2119);border-color:#d19662;color:#f1c197}
.player-box{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.pname{font-family:Georgia,'Times New Roman',serif;font-weight:900;font-size:1.2rem;letter-spacing:.3px;color:#fff}
.badges{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-right:10px}
.badge{background:#0a0f1a;border:1px solid #2a3546;border-radius:999px;padding:6px 12px;font-weight:800;color:#dfe7f4}

/* =================== Cartoon Masters Scoreboard (new) ======================= */
.board-wrap{margin-top:.6rem}
.board-card{border:4px solid #1f2630;border-radius:16px;background:#0b1322;box-shadow:0 10px 26px rgba(0,0,0,.3);padding:8px}
.board-top{display:flex;align-items:center;justify-content:center;background:#10233d;color:#fff;border:2px solid #1c2b43;border-radius:12px;padding:6px 10px;margin:6px}
.board-title{font-family:Georgia,'Times New Roman',serif;font-weight:900;letter-spacing:.8px}
.board-sub{font-weight:800;background:#183a68;border:1px solid #0e2749;border-radius:999px;padding:2px 8px;margin-left:10px}
.board-body{position:relative}
.thru-left,.thru-right{position:absolute;top:48px;width:68px;height:calc(100% - 60px);background:#10233d;color:#e3edf7;border:2px solid #1c2b43;border-radius:10px;display:flex;align-items:flex-start;justify-content:center;padding-top:8px;font-weight:900}
.thru-left{left:-74px} .thru-right{right:-74px}
.x-scroll{overflow-x:auto; -webkit-overflow-scrolling:touch;}
.board{min-width:980px; display:grid; grid-template-columns: 160px repeat(18, 40px) 68px; gap:6px; align-items:center; padding:8px;}
.cell{background:#0f1420;border:1px solid #1f2630;border-radius:10px;color:#e6eef8;display:flex;align-items:center;justify-content:center;height:36px;font-weight:900}
.head{background:#123055;border-color:#1e3f6b;color:#fff;}
.name{justify-content:flex-start;padding:0 10px;font-family:Georgia,'Times New Roman',serif;font-weight:900;letter-spacing:.3px}
.tot{width:100%}
.pip{width:18px;height:18px;border-radius:50%;background:radial-gradient(circle at 35% 35%, #9bffbe, #19c37d);box-shadow:0 0 0 2px #0a7c49 inset, 0 2px 0 #083b26}
.empty{opacity:.25}

/* Mobile tweaks */
@media (max-width: 520px){
  .block-container{ padding: .4rem .5rem; }
  .golf-hero{padding:.6rem .7rem}
  .pname{font-size:1.05rem}
  .masters-wrap{padding:6px}
  .board{min-width:720px; grid-template-columns: 140px repeat(18, 34px) 60px;}
  .cell{height:32px}
  .thru-left,.thru-right{display:none} /* hide side panels on very small screens */
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

# =============================== STATE ========================================
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
            players=[], teams={"Team A": [], "Team B": []}, points={},
            current_hole=1, hole_winners=[None]*18, history=[],
            show_results=False, started_at=time.time(), fx_armed=False,
            fx_tick=0, show_end_confirm=False, toast=None,
            plus1_players=[], plus1_until=0.0,
        )
    upgrade_state()

def upgrade_state() -> None:
    rs = st.session_state.rs
    for k, v in dict(history=[], show_results=False, hole_winners=[None]*18,
                     current_hole=1, teams={"Team A": [], "Team B": []},
                     points={}, players=[], started_at=time.time(), fx_armed=False,
                     fx_tick=0, show_end_confirm=False, toast=None,
                     plus1_players=[], plus1_until=0.0).items():
        if not hasattr(rs, k):
            setattr(rs, k, v)
    if len(rs.hole_winners) != 18: rs.hole_winners = (rs.hole_winners + [None]*18)[:18]

# =============================== HELPERS ======================================
def sanitize_players(inputs: List[str]) -> List[str]:
    seen=set(); out=[]
    for raw in inputs:
        name = raw.strip()
        if name and name.lower() not in seen:
            out.append(name); seen.add(name.lower())
    if len(out) not in (2,4): raise ValueError("Enter exactly 2 or 4 distinct names.")
    return out

def random_pair(players: List[str]) -> Dict[str, List[str]]:
    sh = players[:]; random.shuffle(sh); split = len(sh)//2 if len(sh)>=2 else 1
    return {"Team A": sh[:split], "Team B": sh[split:]}

def set_players(players: List[str]) -> None:
    rs: RoundState = st.session_state.rs
    rs.players = players
    rs.points = {p: rs.points.get(p, 0) for p in players}
    if not any(rs.teams.values()):
        rs.teams = random_pair(players)

def ordinal(n:int)->str:
    return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def compute_ties(players: List[str], points: Dict[str,int]) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,int]]:
    ordered = sorted(((p, points.get(p,0)) for p in players), key=lambda kv:(-kv[1], kv[0]))
    labels:Dict[str,str]={}; notes:Dict[str,str]={}; ranks:Dict[str,int]={}
    i=0; rank=0
    while i < len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][1]==ordered[i][1]: j+=1
        group=[p for p,_ in ordered[i:j]]; rank+=1
        if len(group)>1:
            for p in group:
                labels[p]=f"T-{rank}"; ranks[p]=rank
                others=[x for x in group if x!=p]
                notes[p]=f"tied with {', '.join(others)} for {ordinal(rank)}"
        else:
            labels[group[0]]=str(rank); ranks[group[0]]=rank; notes[group[0]]=""
        i=j
    return labels, notes, ranks

def record_winner(team_name: str) -> None:
    rs: RoundState = st.session_state.rs
    idx = rs.current_hole - 1
    if idx<0 or idx>17 or rs.hole_winners[idx] is not None: return
    winners = [p for p in rs.teams.get(team_name, []) if p in rs.players]
    for p in winners: rs.points[p] = rs.points.get(p, 0) + 1
    rs.hole_winners[idx]=team_name
    rs.history.append({"hole": rs.current_hole, "Team A": rs.teams["Team A"][:], "Team B": rs.teams["Team B"][:], "Winner": team_name})
    rs.toast = f"🏆 Hole {rs.current_hole}: {team_name} – +1 to " + ", ".join(winners) if winners else f"🏆 Hole {rs.current_hole}: {team_name}"
    rs.plus1_players = winners; rs.plus1_until = time.time()+1.2
    if rs.current_hole==18: rs.show_results=True
    else: rs.current_hole += 1; rs.teams = random_pair(rs.players)

def undo_last_hole() -> None:
    rs: RoundState = st.session_state.rs
    if not rs.history: return
    last = rs.history.pop()
    hole = last["hole"]; winner_team = last["Winner"]
    losers=[]
    for p in last[winner_team]:
        if p in rs.players: rs.points[p]=max(0, rs.points.get(p,0)-1); losers.append(p)
    rs.hole_winners[hole-1]=None; rs.current_hole=hole; rs.teams={"Team A":last["Team A"], "Team B":last["Team B"]}
    rs.show_results=False; rs.toast = f"↩️ Undid hole {hole}: removed 1 from " + ", ".join(losers) if losers else f"↩️ Undid hole {hole}"
    rs.plus1_players=[]

def results_df() -> pd.DataFrame:
    rs: RoundState = st.session_state.rs
    df = pd.DataFrame([{"Player":p, "Points":rs.points.get(p,0)} for p in rs.players])
    return df.sort_values(by=["Points","Player"], ascending=[False,True]).reset_index(drop=True)

# ---------- FX (ball only) ----------
def render_fx():
    rs: RoundState = st.session_state.rs
    if not rs.fx_armed: return
    rs.fx_armed=False; rs.fx_tick+=1
    st.markdown(f"""
<div style="position:relative;height:70px;overflow:hidden;margin:.2rem 0 .6rem 0">
  <div style="position:absolute;left:-60px;top:36px;width:16px;height:16px;border-radius:50%;
    background:radial-gradient(circle at 40% 35%, #fff, #dcdcdc);box-shadow:0 2px 0 #bcbcbc;
    animation:drive .9s ease-out forwards, spin .9s linear"></div>
  <div style="position:absolute;left:-60px;top:53px;width:20px;height:6px;border-radius:50%;
    background:rgba(0,0,0,.25);filter:blur(2px);animation:shadowDrive .9s ease-out forwards"></div>
</div>
""", unsafe_allow_html=True)

# ---------- Scoreboard helpers ----------
def build_win_map(players: List[str], history: List[Dict]) -> Dict[str, List[bool]]:
    win_map = {p: [False]*18 for p in players}
    for entry in history:
        h = entry["hole"]
        if not (1 <= h <= 18): continue
        winners = entry.get(entry.get("Winner",""), [])
        for p in winners:
            if p in win_map: win_map[p][h-1] = True
    return win_map

# ========================= IMAGE (results poster) =============================
def _font(size:int):
    try: return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception: return ImageFont.load_default()

def make_podium_image(df: pd.DataFrame) -> bytes:
    W,H=900,520
    img=Image.new("RGB",(W,H),(8,100,40)); d=ImageDraw.Draw(img)
    d.rectangle([(0,H-180),(W,H)], fill=(14,122,36))
    slots=[{"x":W//2-120,"w":240,"h":210,"c":(255,215,0)},
           {"x":W//2-320,"w":220,"h":150,"c":(192,192,192)},
           {"x":W//2+120,"w":220,"h":110,"c":(205,127,50)}]
    d.rounded_rectangle([(30,20),(W-30,90)],16, fill=(255,255,255))
    d.text((50,35),"Round Results", fill=(10,80,20), font=_font(36))
    base = H-180
    for i,s in enumerate(slots):
        x0=s["x"]; y0=base-s["h"]
        d.rounded_rectangle([(x0,y0),(x0+s["w"],base)],12, fill=s["c"])
        if i < len(df):
            row=df.iloc[i]; d.text((x0+16,y0+18), f"{row['Player']} • {int(row['Points'])} pts", fill=(0,0,0), font=_font(24))
    buf=io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

# ============================== UI PARTS ======================================
def chip(player:str, pts:int, place:str, note:str, plus1:bool):
    tied_html = f"<span class='tie-badge'>({note})</span>" if note else ""
    plus_html = "<span class='plus1'>+1</span>" if plus1 else ""
    st.markdown(f"<div class='player-chip'>🏁 <span class='player-name'>{player}</span> {tied_html} "
                f"<span class='player-meta'>Current Place: {place}</span> "
                f"<span class='player-meta'>Total Points: {pts}</span>{plus_html}</div>", unsafe_allow_html=True)

def team_block(title:str, players:List[str], points:Dict[str,int], labels:Dict[str,str], notes:Dict[str,str], active:set):
    st.markdown(f"#### {title}")
    with st.container(border=True):
        for p in players: chip(p, points.get(p,0), labels.get(p,"—"), notes.get(p,""), p in active)

def _medal_class(rank:int)->str:
    return "gold" if rank==1 else "silver" if rank==2 else "bronze" if rank==3 else ""

def render_masters_leaderboard(players:List[str], points:Dict[str,int], labels:Dict[str,str], notes:Dict[str,str], ranks:Dict[str,int], active:set):
    ordered = sorted(((p, points.get(p,0)) for p in players), key=lambda kv:(-kv[1], kv[0]))
    st.markdown("<div class='masters-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='mast-head'><div class='mast-title'>LEADERBOARD</div><div class='mast-pill'>Augusta Style</div></div>", unsafe_allow_html=True)
    for p,pts in ordered:
        rank=ranks.get(p,99); medal=_medal_class(rank)
        tied_html = f"<span class='tie-note'>{notes.get(p,'')}</span>" if notes.get(p) else ""
        plus_html = "<span class='lb-plus1'>+1</span>" if p in active else ""
        st.markdown(f"""
<div class="mrow">
  <div class="mcell"><div class="rank-medal {medal}">#{rank}</div></div>
  <div class="mcell"><div class="player-box"><span class="pname">{p}</span> {tied_html} {plus_html}</div></div>
  <div class="mcell"><div class="badges">
      <span class="badge">Current Place: <b>{labels.get(p,'—')}</b></span>
      <span class="badge">Total Points: <b>{pts}</b></span>
  </div></div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Cartoon Masters Scoreboard ----------
def render_cartoon_scoreboard(players: List[str], points: Dict[str,int], win_map: Dict[str, List[bool]], thru: int):
    st.subheader("Scoreboard")
    st.markdown("<div class='board-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='board-card'>", unsafe_allow_html=True)
    st.markdown("<div class='board-top'><div class='board-title'>LEADERS</div>"
                "<div class='board-sub'>Cartoon Masters Board</div></div>", unsafe_allow_html=True)
    # Side panels (hidden on small screens)
    st.markdown(f"<div class='board-body'><div class='thru-left'>THRU {min(thru,9)}</div>"
                f"<div class='thru-right'>THRU {thru}</div>", unsafe_allow_html=True)
    st.markdown("<div class='x-scroll'>", unsafe_allow_html=True)

    # Header row
    header = ["<div class='cell head name'>PLAYER</div>"] + \
             [f"<div class='cell head'>{i}</div>" for i in range(1,19)] + \
             ["<div class='cell head tot'>TOT</div>"]
    st.markdown("<div class='board'>" + "".join(header) + "</div>", unsafe_allow_html=True)

    # Rows
    ordered = sorted(players, key=lambda p: (-points.get(p,0), p))
    for p in ordered:
        cells = [f"<div class='cell name'>{p}</div>"]
        wins = win_map.get(p, [False]*18)
        for won in wins:
            cells.append(f"<div class='cell'>{'<div class=\"pip\"></div>' if won else '<div class=\"pip empty\"></div>'}</div>")
        cells.append(f"<div class='cell tot'>{points.get(p,0)}</div>")
        st.markdown("<div class='board'>" + "".join(cells) + "</div>", unsafe_allow_html=True)

    st.markdown("</div></div></div></div>", unsafe_allow_html=True)  # close containers

# =============================== APP ==========================================
def main():
    st.set_page_config(page_title="Golf Random Teams", page_icon="⛳", layout="wide")
    init_state()
    st.markdown(GOLF_CSS, unsafe_allow_html=True)
    rs: RoundState = st.session_state.rs

    # Toast
    if rs.toast:
        st.toast(rs.toast); rs.toast=None

    # Ball animation
    render_fx()

    # Blocking confirm pane
    if rs.show_end_confirm:
        st.markdown("<div class='confirm-wrap'>", unsafe_allow_html=True)
        with st.container(border=False):
            st.markdown("<div class='confirm-pane'>"
                        "<div class='confirm-title'>End Round?</div>"
                        "<div class='confirm-text'>This will <b>delete all holes, teams, and points</b> for this round.</div>"
                        "</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("No, keep playing"): rs.show_end_confirm=False; st.rerun()
            with c2:
                if st.button("🛑 End Round", type="primary"):
                    st.session_state.pop("rs", None); init_state(); st.success("Round reset. Enter player names to begin."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Results
    if rs.show_results:
        st.success("Round complete! 🎉 Final results below.")
        df_res = results_df()
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        png = make_podium_image(df_res)
        st.download_button("🖼️ Save Results Poster (PNG)", data=png, file_name="golf_results.png", mime="image/png")
        st.download_button("📄 Save Standings (CSV)", data=df_res.to_csv(index=False).encode("utf-8"),
                           file_name="golf_standings.csv", mime="text/csv")

    # Header
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}'
                f'<div><div class="golf-badge">Random Teams • Mobile-first • Masters Board</div>'
                f'<div style="opacity:.9">+1 to each winner • Auto-randomize after each hole • Undo & Confirm Reset</div></div></div>',
                unsafe_allow_html=True)

    # Players (lock after Hole 1 result)
    names_locked = (rs.hole_winners[0] is not None)
    with st.container(border=True):
        if not names_locked:
            c1,c2,c3,c4 = st.columns(4)
            inputs = [c1.text_input("Player 1", value=(rs.players[0] if len(rs.players)>0 else "")),
                      c2.text_input("Player 2", value=(rs.players[1] if len(rs.players)>1 else "")),
                      c3.text_input("Player 3 (optional)", value=(rs.players[2] if len(rs.players)>2 else "")),
                      c4.text_input("Player 4 (optional)", value=(rs.players[3] if len(rs.players)>3 else ""))]
            a,b = st.columns([1,1])
            with a:
                if st.button("✅ Set / Update Players", use_container_width=True):
                    try: set_players(sanitize_players(inputs)); st.success("Players updated.")
                    except ValueError as e: st.error(str(e))
            with b:
                if st.button("🎲 Randomize Teams now", use_container_width=True, disabled=not rs.players):
                    rs.teams = random_pair(rs.players); rs.fx_armed=True; st.rerun()
        else:
            roster = " • ".join(rs.players) if rs.players else "—"
            st.markdown("**Players are locked for this round (after Hole 1). Use _End Round_ to change.**")
            st.markdown(f"Current players: **{roster}**")

    if not rs.players:
        st.info("Enter 2 or 4 names above to begin.")
        return

    # Tie labels + ranks; +1 window
    labels, notes, ranks = compute_ties(rs.players, rs.points)
    active_plus1 = set(rs.plus1_players) if time.time() < rs.plus1_until else set()
    if rs.plus1_players and time.time() >= rs.plus1_until: rs.plus1_players=[]

    # Teams
    cA,cB = st.columns(2)
    with cA: team_block("Team A", rs.teams["Team A"], rs.points, labels, notes, active_plus1)
    with cB: team_block("Team B", rs.teams["Team B"], rs.points, labels, notes, active_plus1)

    # Sticky winner controls (mobile friendly)
    with st.container():
        st.markdown("<div class='sticky-ctrl'>", unsafe_allow_html=True)
        st.subheader(f"Hole {rs.current_hole} / 18 • Record Winner")
        disabled = rs.hole_winners[rs.current_hole-1] is not None if 1 <= rs.current_hole <= 18 else True
        wA, wB, actions = st.columns([1, 1, 2])
        with wA:
            if st.button("🏆 Team A won", use_container_width=True, disabled=disabled or rs.show_results):
                record_winner("Team A"); rs.fx_armed=True; st.rerun()
        with wB:
            if st.button("🏆 Team B won", use_container_width=True, disabled=disabled or rs.show_results):
                record_winner("Team B"); rs.fx_armed=True; st.rerun()
        with actions:
            if st.button("↩️ Undo Last Hole", use_container_width=True, disabled=not rs.history):
                undo_last_hole(); rs.fx_armed=True; st.rerun()
            st.metric("Holes recorded", sum(1 for w in rs.hole_winners if w is not None))
        st.markdown("</div>", unsafe_allow_html=True)

    # Fancy leaderboard (Masters-style cards)
    render_masters_leaderboard(rs.players, rs.points, labels, notes, ranks, active_plus1)

    # Cartoon Masters scoreboard (swipeable on mobile)
    win_map = build_win_map(rs.players, rs.history)
    render_cartoon_scoreboard(rs.players, rs.points, win_map, thru=rs.current_hole-1)

    # Hole log table (collapsible feel: show only when there is history)
    st.subheader("Hole Log")
    if rs.history:
        log_df = pd.DataFrame(rs.history)[["hole","Team A","Team B","Winner"]].rename(columns={"hole":"Hole"})
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No holes recorded yet.")

    st.markdown("---")
    if st.button("🛑 End Round", type="primary", use_container_width=True):
        rs.show_end_confirm = True; st.rerun()

if __name__ == "__main__":
    main()
