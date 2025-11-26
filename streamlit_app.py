# file: streamlit_app.py
import base64
import io
import math
import random
import struct
import time
import wave
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# -------------------------- THEME & FX CSS ------------------------------------
GOLF_CSS = """
<style>
:root { --golf-green:#0b7a24; --golf-dark:#064a15; --ink:#0b0e11; }
section.main { background: radial-gradient(1200px 600px at 10% -10%, #ffffff, #eaf7e7) }
.block-container { padding-top: .8rem; }

/* Header */
.golf-hero{padding:.8rem 1rem;border-radius:12px;background:linear-gradient(135deg,var(--golf-green),var(--golf-dark));color:#fff;display:flex;align-items:center;gap:14px}
.golf-badge{background:#ffffff22;padding:6px 10px;border-radius:10px;font-weight:800}

/* Team cards + chips */
.team-card{border:3px solid var(--golf-green);border-radius:16px;padding:14px;background:#fff;box-shadow:0 2px 10px rgba(0,0,0,.08)}
.player-chip{display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:14px;border:2px solid var(--golf-green);background:#fff;margin:6px 6px;flex-wrap:wrap}
.player-name{font-weight:900;font-size:1.05rem;color:#0c0c0c}
.player-meta{font-weight:800;color:#0b7a24;background:#eaf7e7;border:1px solid #bde0c2;border-radius:999px;padding:4px 10px}
.tie-badge{font-weight:900;background:#fff3bf;color:#8a6700;border:1px solid #e3c200;border-radius:8px;padding:2px 6px}

/* Leaderboard */
.leaderboard-wrap{margin-top:.2rem}
.lb-row{border-radius:18px;padding:14px 16px;background:#0f1420;border:2px solid #1f2630;box-shadow:0 6px 22px rgba(0,0,0,.25);color:#eef2f7;margin-bottom:10px}
.lb-name{font-family:'Trebuchet MS','Segoe UI',system-ui,sans-serif;font-weight:900;font-size:1.2rem;color:#fff;display:flex;align-items:center;gap:8px}
.lb-lines{display:flex;gap:12px;margin-top:6px;flex-wrap:wrap}
.lb-tag{background:#0a0f1a;border:1px solid #2a3546;border-radius:999px;padding:6px 12px;font-weight:800;color:#dfe7f4}

/* Buttons */
div.stButton > button[kind="primary"]{ background:#c0392b;border-color:#8e2a1b;color:#fff; }
div.stButton > button[kind="primary"]:hover{ background:#a33224; }

/* FX */
.fx-area{position:relative;height:80px;overflow:hidden;margin:.2rem 0 .6rem 0}
.fx-ball{position:absolute;left:-60px;top:40px;width:18px;height:18px;border-radius:50%;background:radial-gradient(circle at 40% 35%, #fff, #dcdcdc);box-shadow:0 2px 0 #bcbcbc;animation:drive .9s ease-out forwards, spin .9s linear}
.fx-shadow{position:absolute;left:-60px;top:59px;width:22px;height:6px;border-radius:50%;background:rgba(0,0,0,.25);filter:blur(2px);animation:shadowDrive .9s ease-out forwards}
@keyframes drive{0%{transform:translate(0,0)} 30%{transform:translate(220px,-36px)} 70%{transform:translate(700px,-6px)} 100%{transform:translate(1100px,6px)}}
@keyframes shadowDrive{0%{transform:translate(0,0) scale(.6)} 30%{transform:translate(220px,6px) scale(1)} 100%{transform:translate(1100px,8px) scale(.8)}}
@keyframes spin{0%{transform:rotate(0)} 100%{transform:rotate(420deg)}}

/* Confirm overlay */
.confirm-mask{position:fixed;inset:0;background:rgba(0,0,0,.45);backdrop-filter:blur(2px);z-index:9998}
.confirm-card{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:9999;background:#111827;color:#fff;border:2px solid #374151;border-radius:14px;padding:16px 18px;min-width:320px;box-shadow:0 18px 60px rgba(0,0,0,.5)}
.confirm-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:12px}
.confirm-danger{background:#dc2626;border:1px solid #991b1b;color:#fff;border-radius:8px;padding:8px 12px;font-weight:800}
.confirm-cancel{background:#111827;border:1px solid #374151;color:#e5e7eb;border-radius:8px;padding:8px 12px;font-weight:800}
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
    show_end_confirm: bool  # popup flag

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
    """+1 to each winner; log; advance; re-roll; results at 18."""
    rs: RoundState = st.session_state.rs
    idx = rs.current_hole - 1
    if idx < 0 or idx > 17 or rs.hole_winners[idx] is not None:
        return
    for p in rs.teams.get(team_name, []):
        rs.points[p] = rs.points.get(p, 0) + 1
    rs.hole_winners[idx] = team_name
    rs.history.append({
        "hole": rs.current_hole,
        "Team A": rs.teams["Team A"][:],
        "Team B": rs.teams["Team B"][:],
        "Winner": team_name,
    })
    if rs.current_hole == 18:
        rs.show_results = True
    else:
        rs.current_hole += 1
        rs.teams = random_pair(rs.players)

def undo_last_hole() -> None:
    """Reverse the most recent recorded hole (points, winner, teams, hole)."""
    rs: RoundState = st.session_state.rs
    if not rs.history:
        return
    last = rs.history.pop()
    hole = last["hole"]
    winner_team = last["Winner"]
    for p in last[winner_team]:
        rs.points[p] = max(0, rs.points.get(p, 0) - 1)
    rs.hole_winners[hole - 1] = None
    rs.current_hole = hole
    rs.teams = {"Team A": last["Team A"], "Team B": last["Team B"]}
    rs.show_results = False

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

def tied_rank_labels(points: Dict[str, int]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Dense ranks + human labels with ties ('T-1')."""
    pts_to_players: Dict[int, List[str]] = {}
    for p, pts in points.items():
        pts_to_players.setdefault(pts, []).append(p)
    unique_pts = sorted(pts_to_players.keys(), reverse=True)
    labels: Dict[str, str] = {}
    ranks: Dict[str, int] = {}
    rank = 0
    for pts in unique_pts:
        rank += 1
        group = sorted(pts_to_players[pts])
        tied = len(group) > 1
        for p in group:
            labels[p] = f"T-{rank}" if tied else str(rank)
            ranks[p] = rank
    return labels, ranks

def combo_stats() -> pd.DataFrame:
    rs: RoundState = st.session_state.rs
    if len(rs.players) < 2: return pd.DataFrame(columns=["Pair", "Times Teamed", "% of 18"])
    counts: Dict[Tuple[str, str], int] = {tuple(sorted(pair)): 0 for pair in combinations(rs.players, 2)}
    for entry in rs.history:
        for team in ("Team A", "Team B"):
            for a, b in combinations(sorted(entry[team]), 2):
                counts[(a, b)] += 1
    rows = [{"Pair": f"{a} + {b}", "Times Teamed": n, "% of 18": round(100 * n / 18, 1)}
            for (a, b), n in counts.items()]
    return pd.DataFrame(rows).sort_values(by=["Times Teamed", "Pair"], ascending=[False, True])

# ------------------------------ FX: Driver WAP --------------------------------
def _make_driver_wap_wav_bytes(duration_s: float = 0.28, sr: int = 44100) -> bytes:
    n = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        lp = 0.0
        for i in range(n):
            t = i / sr
            attack = min(1.0, t / 0.003)
            env_crack = math.exp(-t / 0.025)
            env_thump = math.exp(-t / 0.12)
            env_noise = math.exp(-t / 0.09)
            f = 2200.0 - 1000.0 * t
            crack = math.sin(2 * math.pi * f * t) * env_crack
            thump = (0.75 * math.sin(2 * math.pi * 120 * t) + 0.25 * math.sin(2 * math.pi * 60 * t)) * env_thump
            raw = (random.random() * 2 - 1)
            lp = lp + 0.2 * (raw - lp)
            hp = raw - lp
            whoosh = hp * env_noise
            sample = math.tanh(1.6 * (attack * (0.55 * crack + 0.6 * thump + 0.2 * whoosh)))
            wf.writeframes(struct.pack("<h", int(max(-1, min(1, sample)) * 32767)))
    return buf.getvalue()

_DRIVER_WAP_B64 = base64.b64encode(_make_driver_wap_wav_bytes()).decode("ascii")

def render_fx():
    rs: RoundState = st.session_state.rs
    if not rs.fx_armed: return
    rs.fx_armed = False
    rs.fx_tick += 1
    st.markdown(f"""
<div class="fx-area" id="fx{rs.fx_tick}">
  <div class="fx-ball"></div>
  <div class="fx-shadow"></div>
</div>
<audio autoplay style="display:none">
  <source src="data:audio/wav;base64,{_DRIVER_WAP_B64}" type="audio/wav">
</audio>
""", unsafe_allow_html=True)

# ------------------------------ IMAGE (results poster) ------------------------
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
    try: return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception: return ImageFont.load_default()

# ------------------------------ UI PARTS --------------------------------------
def chip_with_editor(player: str, points: int, place_label: str, tied: bool) -> None:
    col_chip, col_plus, col_minus, col_num = st.columns([3, 1, 1, 1.3])
    with col_chip:
        tied_html = f"<span class='tie-badge'>(tied)</span>" if tied else ""
        st.markdown(
            f"<div class='player-chip'>🏁 <span class='player-name'>{player}</span> {tied_html} "
            f"<span class='player-meta'>Current Place: {place_label}</span> "
            f"<span class='player-meta'>Total Points: {points}</span></div>",
            unsafe_allow_html=True,
        )
    with col_plus:
        if st.button("➕", key=f"inc_{player}", use_container_width=True):
            adjust_point(player, +1); st.session_state.rs.fx_armed = True; st.rerun()
    with col_minus:
        if st.button("➖", key=f"dec_{player}", use_container_width=True):
            adjust_point(player, -1); st.session_state.rs.fx_armed = True; st.rerun()
    with col_num:
        new_val = st.number_input(f"{player} pts", min_value=0, max_value=99, value=int(points),
                                  key=f"num_{player}", label_visibility="collapsed")
        if new_val != points:
            set_point(player, new_val); st.session_state.rs.fx_armed = True; st.rerun()

def team_block_editable(team_name: str, players: List[str], points: Dict[str, int], labels: Dict[str, str]) -> None:
    st.markdown(f"#### {team_name}")
    with st.container(border=True):
        for p in players:
            chip_with_editor(p, points.get(p, 0), labels.get(p, "—"), labels.get(p, "").startswith("T-"))

def render_leaderboard(points: Dict[str, int], labels: Dict[str, str]) -> None:
    ordered = [(p, pts) for p, pts in sorted(points.items(), key=lambda kv: (-kv[1], kv[0]))]
    st.subheader("Leaderboard")
    st.markdown("<div class='leaderboard-wrap'>", unsafe_allow_html=True)
    for p, pts in ordered:
        lab = labels.get(p, "—"); tied = lab.startswith("T-")
        tied_html = "<span class='tie-badge'>(tied)</span>" if tied else ""
        st.markdown(
            f"""
<div class="lb-row">
  <div class="lb-name">{p} {tied_html}</div>
  <div class="lb-lines">
    <span class="lb-tag">Current Place: <b>{lab}</b></span>
    <span class="lb-tag">Total Points: <b>{pts}</b></span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------ APP -------------------------------------------
def main():
    st.set_page_config(page_title="Golf Random Teams", page_icon="⛳", layout="wide")
    init_state()
    st.markdown(GOLF_CSS, unsafe_allow_html=True)

    rs: RoundState = st.session_state.rs

    # FX from previous interaction
    render_fx()

    # Results on 18
    if rs.show_results:
        st.success("Round complete! 🎉 Final results below.")
        df_res = results_df()
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        png = make_podium_image(df_res)
        st.download_button("🖼️ Save Results Poster (PNG)", data=png, file_name="golf_results.png", mime="image/png")
        st.download_button("📄 Save Standings (CSV)", data=df_res.to_csv(index=False).encode("utf-8"),
                           file_name="golf_standings.csv", mime="text/csv")
        if rs.history:
            st.download_button("📄 Save Hole Log (CSV)",
                               data=pd.DataFrame(rs.history).to_csv(index=False).encode("utf-8"),
                               file_name="golf_hole_log.csv", mime="text/csv")

    # Header
    st.markdown(f'<div class="golf-hero">{GOLF_SVG}'
                f'<div><div class="golf-badge">Random Teams • Score • Live Leaderboard</div>'
                f'<div style="opacity:.9">Auto randomize after each hole • Names lock after Hole 1 • Undo & Confirm Reset</div></div></div>',
                unsafe_allow_html=True)

    # Player inputs (hide after Hole 1 is recorded)
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
                        set_players(players); st.success("Players updated.")
            with b2:
                if st.button("🎲 Randomize Teams now", use_container_width=True, disabled=not rs.players):
                    rs.teams = random_pair(rs.players); rs.fx_armed = True; st.rerun()
        else:
            st.markdown("**Players are locked for this round (after Hole 1). Use _End Round_ to change.**")
            roster = " • ".join(rs.players) if rs.players else "—"
            st.markdown(f"Current players: **{roster}**")

    if not rs.players:
        st.info("Enter 2 or 4 names above to begin.")
    else:
        # Tie-aware labels from current points
        labels, _ = tied_rank_labels(rs.points)

        # Teams + inline editors
        colA, colB = st.columns(2)
        with colA: team_block_editable("Team A", rs.teams["Team A"], rs.points, labels)
        with colB: team_block_editable("Team B", rs.teams["Team B"], rs.points, labels)

        st.divider()
        st.subheader(f"Hole {rs.current_hole} / 18 • Record Winner (auto-randomizes next)")
        disabled = rs.hole_winners[rs.current_hole-1] is not None if 1 <= rs.current_hole <= 18 else True
        wA, wB, actions = st.columns([1, 1, 2])
        with wA:
            if st.button("🏆 Team A won", use_container_width=True, disabled=disabled or rs.show_results):
                record_winner("Team A"); rs.fx_armed = True; st.rerun()
        with wB:
            if st.button("🏆 Team B won", use_container_width=True, disabled=disabled or rs.show_results):
                record_winner("Team B"); rs.fx_armed = True; st.rerun()
        with actions:
            if st.button("↩️ Undo Last Hole", use_container_width=True, disabled=not rs.history):
                undo_last_hole(); st.rerun()
            st.metric("Holes recorded", sum(1 for w in rs.hole_winners if w is not None))

        # Leaderboard
        render_leaderboard(rs.points, labels)

        # Hole log
        st.subheader("Hole Log")
        if rs.history:
            log_df = pd.DataFrame(rs.history)[["hole", "Team A", "Team B", "Winner"]].rename(columns={"hole": "Hole"})
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No holes recorded yet.")

    # --- End Round with confirmation ------------------------------------------
    st.markdown("---")
    if st.button("🛑 End Round", type="primary", use_container_width=True):
        rs.show_end_confirm = True
        st.experimental_rerun()

    if rs.show_end_confirm:
        # modal-like confirm
        st.markdown("<div class='confirm-mask'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='confirm-card'><div style='font-weight:900;font-size:1.1rem'>End Round?</div>"
            "<div style='margin-top:8px'>This will <b>delete all holes, teams, and points</b> for this round.</div>"
            "<div class='confirm-actions'>"
            "<button class='confirm-cancel' onclick='window.location.reload()' disabled style='opacity:.01;position:absolute;left:-9999px'>noop</button>"
            "</div></div>",  # dummy button to avoid HTML-only
            unsafe_allow_html=True,
        )
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("No, keep playing", key="cancel_end"):
                rs.show_end_confirm = False
                st.experimental_rerun()
        with c2:
            if st.button("Yes, end round (erase all)", key="confirm_end"):
                st.session_state.pop("rs", None)
                init_state()
                st.success("Round reset. Enter player names to begin.")
                st.experimental_rerun()

if __name__ == "__main__":
    main()
