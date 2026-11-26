"""
============================================================
 spam_alert_app.py - Two-account GNN spam alert demo
============================================================
Left half  = ORIGINAL account (receiver) : profile, inbox, alerts, network
Right half = SENDER account              : fake account builder (or real follower /
             mutual / stranger from the dataset), message box, live risk analysis

Run order:
    pip install streamlit pandas numpy scipy scikit-learn plotly joblib
    python train_gnn.py            (once)
    streamlit run spam_alert_app.py
"""
import html
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_engine import RiskEngine, age_text, estimate_signup

st.set_page_config(page_title="GNN Spam Alert", page_icon="🚨", layout="wide")

BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.joblib")
LEVEL_COLOR = {"LOW": "#21a366", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
BLUE, RED, GREEN, AMBER, PURPLE = "#3b82f6", "#ef4444", "#21a366", "#f59e0b", "#a855f7"

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 100%;}
.hero {font-size:30px;font-weight:800;margin-bottom:0;}
.hero-sub {color:#9ca3af;margin-bottom:12px;}
.panel-title {font-size:17px;font-weight:800;letter-spacing:.6px;padding:10px 16px;border-radius:12px;
  color:#fff;margin-bottom:12px;text-align:center;}
.card {background:linear-gradient(145deg,#111827,#1f2937);color:#f9fafb;border-radius:18px;
  padding:18px 20px;margin-bottom:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);}
.card .row {display:flex;align-items:center;gap:14px;}
.avatar {width:58px;height:58px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:30px;flex-shrink:0;}
.handle {font-size:20px;font-weight:700;}
.uid {color:#9ca3af;font-size:13px;}
.badge {margin-left:auto;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;color:#fff;}
.bio {color:#d1d5db;font-size:14px;margin:10px 0 12px 0;}
.stats {display:flex;justify-content:space-between;gap:8px;text-align:center;}
.stats div {flex:1;background:rgba(255,255,255,.06);border-radius:12px;padding:8px 4px;}
.stats b {display:block;font-size:20px;}
.stats span {font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;}
.alert-banner {background:linear-gradient(135deg,#7f1d1d,#dc2626);color:#fff;border-radius:16px;padding:18px 22px;
  margin-bottom:12px;animation:pulse 1.4s infinite, shake .6s 1;}
.warn-banner {background:linear-gradient(135deg,#78350f,#d97706);color:#fff;border-radius:16px;padding:16px 22px;margin-bottom:12px;}
.ok-banner {background:linear-gradient(135deg,#064e3b,#059669);color:#fff;border-radius:16px;padding:14px 22px;margin-bottom:12px;}
.alert-banner h3,.warn-banner h3,.ok-banner h3 {margin:0 0 4px 0;color:#fff;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(239,68,68,.75);}70%{box-shadow:0 0 0 22px rgba(239,68,68,0);}100%{box-shadow:0 0 0 0 rgba(239,68,68,0);}}
@keyframes shake {0%,100%{transform:translateX(0);}20%{transform:translateX(-8px);}40%{transform:translateX(8px);}60%{transform:translateX(-5px);}80%{transform:translateX(5px);}}
.msg {border-radius:12px;padding:10px 14px;margin-bottom:8px;background:#111827;color:#f3f4f6;border-left:6px solid #21a366;}
.msg.MEDIUM {border-left-color:#f59e0b;} .msg.HIGH {border-left-color:#ef4444;background:#2a1215;}
.msg .top {display:flex;justify-content:space-between;font-size:12px;color:#9ca3af;margin-bottom:3px;}
.pill {padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;color:#fff;}
.reason {background:rgba(148,163,184,.12);border-radius:10px;padding:7px 12px;margin-bottom:6px;font-size:14px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ engine / state
@st.cache_resource(show_spinner="Loading trained GNN model...")
def get_engine():
    return RiskEngine(BUNDLE)


if not os.path.exists(BUNDLE):
    st.error("model_bundle.joblib not found. Run  `python train_gnn.py`  first (once), "
             "then start the app again.")
    st.stop()

eng = get_engine()

PRESETS = {
    "✉️ Normal message": "Hey, are you free for a call tomorrow?",
    "🚨 Spam message": "Congratulations! You have won a free cash prize. Click the link to claim now!",
    "🎭 Polite fake (no spam words)": "Hi! I really like your profile, can we connect and chat?",
}
for k, v in {"inbox": [], "blocked": set(), "last": None, "flash": None, "fake_seed": 1,
             "msg": PRESETS["🚨 Spam message"], "inbox_for": None, "reports": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def set_preset():
    st.session_state.msg = PRESETS[st.session_state.preset]


def send_message():
    s = st.session_state.cur_sender
    msg = st.session_state.msg.strip()
    if not msg:
        st.session_state.flash = ("warn", "Type a message first.")
        return
    if s["uid"] in st.session_state.blocked:
        st.session_state.flash = ("blocked", f"🚫 Not delivered - @{s['handle']} is blocked by the receiver.")
        return
    t = eng.text_prob(msg)
    res = eng.combine(s["gnn_p"], t, s["age_days"], s["relation"], eng.keyword_hits(msg))
    res.update(uid=s["uid"], handle=s["handle"], text=msg, time=datetime.now().strftime("%H:%M:%S"),
               is_fake=s["is_fake"], handled=None, x=s["x"])
    st.session_state.inbox.insert(0, res)
    st.session_state.last = res
    st.session_state.flash = ("sent", "Message sent to the receiver's inbox ➜")
    if res["level"] == "HIGH":
        st.toast(f"SPAM ALERT - risk {res['risk']:.0f}/100 from @{s['handle']}", icon="🚨")


def act(kind):
    last = st.session_state.last
    if not last:
        return
    last["handled"] = kind
    if kind == "blocked":
        st.session_state.blocked.add(last["uid"])
    elif kind == "reported":
        st.session_state.reports += 1


def reset_demo():
    st.session_state.inbox, st.session_state.last = [], None
    st.session_state.blocked, st.session_state.flash, st.session_state.reports = set(), None, 0


# ------------------------------------------------------------------ small helpers
def profile_card(handle, uid, followers, following, extra_label, extra_value, age, badge, accent, bio, emoji):
    return f"""
    <div class="card" style="border-top:5px solid {accent}">
      <div class="row">
        <div class="avatar" style="background:{accent}33;border:2px solid {accent}">{emoji}</div>
        <div><div class="handle">@{html.escape(str(handle))}</div><div class="uid">ID {uid}</div></div>
        <div class="badge" style="background:{accent}">{badge}</div>
      </div>
      <div class="bio">{html.escape(bio)}</div>
      <div class="stats">
        <div><b>{followers:,}</b><span>Followers</span></div>
        <div><b>{following:,}</b><span>Following</span></div>
        <div><b>{extra_value}</b><span>{extra_label}</span></div>
        <div><b>{age}</b><span>Account age</span></div>
      </div>
    </div>"""


def radial_fig(center_name, center_color, groups, height=340, key=None):
    """Simple network picture: centre account + coloured rings of neighbours."""
    rng = np.random.default_rng(5)
    total = max(sum(len(g[1]) for g in groups), 1)
    angle_all = np.linspace(0, 2 * np.pi, total, endpoint=False)
    fig, pos = go.Figure(), 0
    ex, ey = [], []
    traces = []
    for label, names, color, size in groups:
        if not len(names):
            continue
        ang = angle_all[pos:pos + len(names)]
        pos += len(names)
        rad = 1 + rng.uniform(-0.18, 0.28, len(names))
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        for xi, yi in zip(x, y):
            ex += [0, xi, None]
            ey += [0, yi, None]
        traces.append(go.Scatter(x=x, y=y, mode="markers", name=f"{label} ({len(names)})",
                                 marker=dict(size=size, color=color, line=dict(width=1, color="#fff")),
                                 text=list(names), hoverinfo="text"))
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="rgba(148,163,184,.45)"),
                             hoverinfo="skip", showlegend=False))
    for t in traces:
        fig.add_trace(t)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=[center_name], textposition="bottom center",
                             name="Account", marker=dict(size=30, color=center_color,
                                                         line=dict(width=3, color="#fff")),
                             textfont=dict(color="#9ca3af", size=11), hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.02, font=dict(color="#9ca3af", size=11)))
    st.plotly_chart(fig, use_container_width=True, key=key)


def names_of(idx, limit):
    idx = np.asarray(idx)[:limit]
    return [f"@user_{eng.node_ids[i]}" if i < eng.n else f"@bot_{i - eng.n}" for i in idx]


def gauge(value, level, key):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=" / 100", font=dict(size=34)),
        title=dict(text=f"Spam risk: {level}", font=dict(size=18, color=LEVEL_COLOR[level])),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=LEVEL_COLOR[level], thickness=0.3),
                   steps=[dict(range=[0, 30], color="rgba(33,163,102,.25)"),
                          dict(range=[30, 60], color="rgba(245,158,11,.25)"),
                          dict(range=[60, 100], color="rgba(239,68,68,.25)")])))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9ca3af"))
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_analysis(res, key):
    st.markdown(f"#### 🔬 Live analysis · @{res['handle']} · {res['time']}")
    a, b = st.columns([1, 1])
    with a:
        gauge(res["risk"], res["level"], f"g{key}")
    with b:
        p = res["parts"]
        steps = [("GNN signal", p["gnn"]), ("Message text", p["text"]), ("Account age", p["age"]),
                 ("Trust discount", -p["discount"])]
        if p["override"] > 0:
            steps.append(("Override", p["override"]))
        fig = go.Figure(go.Waterfall(
            x=[s[0] for s in steps] + ["Final risk"], y=[s[1] for s in steps] + [0],
            measure=["relative"] * len(steps) + ["total"],
            increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=GREEN)),
            totals=dict(marker=dict(color=LEVEL_COLOR[res["level"]]))))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title=dict(text="How the score was built", font=dict(size=14)),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af"),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"w{key}")
    st.markdown("**Why:**")
    for r in res["reasons"]:
        st.markdown(f'<div class="reason">{html.escape(r)}</div>', unsafe_allow_html=True)
    exp = eng.explain(res["x"])
    fig = go.Figure(go.Bar(
        x=[e["suspicion"] for e in exp], y=[e["label"] + f"  ({e['value']})" for e in exp], orientation="h",
        marker=dict(color=[RED if e["flagged"] else (AMBER if e["suspicion"] > 0.5 else GREEN) for e in exp])))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=28, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af"),
                      title=dict(text="Graph features vs normal users (right = more suspicious)", font=dict(size=14)),
                      yaxis=dict(autorange="reversed"), xaxis=dict(zeroline=True))
    st.plotly_chart(fig, use_container_width=True, key=f"f{key}")


# ------------------------------------------------------------------ header + sidebar
st.markdown('<div class="hero">🧠 GNN Social Network Monitor</div>'
            '<div class="hero-sub">Two accounts, one graph: watch the receiver get protected in real time</div>',
            unsafe_allow_html=True)

default_recv = int(eng.node_ids[eng.pick_default_receiver()])
st.sidebar.title("⚙️ Controls")
recv_text = st.sidebar.text_input("Receiver (original account) ID", value=str(default_recv))
digits = "".join(c for c in recv_text if c.isdigit())
r_idx = eng.lookup(int(digits)) if digits else None
if r_idx is None:
    st.sidebar.warning("ID not in the dataset - using the default receiver.")
    r_idx = eng.lookup(default_recv)
r_uid = int(eng.node_ids[r_idx])
st.sidebar.button("🧹 Reset demo (clear inbox & blocks)", on_click=reset_demo, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.metric("Users in network", f"{eng.meta['n_real']:,}")
st.sidebar.metric("Follow connections", f"{eng.meta['n_edges']:,}")
st.sidebar.metric("Synthetic bots used for training", f"{eng.meta['n_bots']:,}")
st.sidebar.caption("Score = 50% GNN + 35% text + 15% account age, minus a capped follower-trust discount.")

if st.session_state.inbox_for != r_idx:
    reset_demo()
    st.session_state.inbox_for = r_idx
st.session_state.recv_idx = r_idx

tab_demo, tab_perf, tab_about = st.tabs(["📡 Live Demo", "📊 Model Performance", "ℹ️ How it works"])

# ================================================================== LIVE DEMO
with tab_demo:
    left, right = st.columns(2, gap="large")

    foll_r, fing_r = eng.followers(r_idx), eng.following(r_idx)
    mut_r = eng.mutuals(r_idx)
    r_signup = estimate_signup(r_uid)
    r_days = (date.today() - r_signup).days

    # ------------------------------------------------ LEFT: ORIGINAL ACCOUNT
    with left:
        st.markdown(f'<div class="panel-title" style="background:{BLUE}">🟦 ORIGINAL ACCOUNT · RECEIVER</div>',
                    unsafe_allow_html=True)
        st.markdown(profile_card(f"user_{r_uid}", r_uid, len(foll_r), len(fing_r), "Mutual", f"{len(mut_r):,}",
                                 age_text(r_days), "VERIFIED ✓", BLUE,
                                 "Genuine account taken from the Twitter network dataset.", "🧑"),
                    unsafe_allow_html=True)

        inbox = st.session_state.inbox
        n_alert = sum(m["level"] == "HIGH" for m in inbox)
        n_warn = sum(m["level"] == "MEDIUM" for m in inbox)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📥 Inbox", len(inbox))
        m2.metric("🚨 Alerts", n_alert)
        m3.metric("⚠️ Cautions", n_warn)
        m4.metric("🚫 Blocked", len(st.session_state.blocked))

        last = st.session_state.last
        if last and not last["handled"]:
            if last["level"] == "HIGH":
                st.markdown(f"""<div class="alert-banner"><h3>🚨 SPAM ALERT - HIGH RISK ({last['risk']:.0f}/100)</h3>
                    Message from <b>@{html.escape(last['handle'])}</b> looks like spam from a
                    {'fake account' if last['gnn_p'] >= 0.5 else 'suspicious sender'}.</div>""",
                            unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                b1.button("🚫 Block user", on_click=act, args=("blocked",), use_container_width=True, key="blk")
                b2.button("📢 Report spam", on_click=act, args=("reported",), use_container_width=True, key="rpt")
                b3.button("✓ Ignore", on_click=act, args=("ignored",), use_container_width=True, key="ign")
            elif last["level"] == "MEDIUM":
                st.markdown(f"""<div class="warn-banner"><h3>⚠️ CAUTION - MEDIUM RISK ({last['risk']:.0f}/100)</h3>
                    Message from <b>@{html.escape(last['handle'])}</b> needs a second look.</div>""",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="ok-banner"><h3>✅ SAFE - no alert ({last['risk']:.0f}/100)</h3>
                    Trusted message from <b>@{html.escape(last['handle'])}</b>
                    ({eng.RELATION_LABEL[last['relation']]}).</div>""", unsafe_allow_html=True)
        elif last and last["handled"]:
            st.info(f"Last alert handled: **{last['handled']}** (@{last['handle']})")

        st.markdown("#### 📥 Inbox")
        if not inbox:
            st.caption("No messages yet - send one from the right side ➜")
        for m in inbox[:8]:
            tag = f" · {m['handled']}" if m["handled"] else ""
            st.markdown(f"""<div class="msg {m['level']}"><div class="top">
                <span>@{html.escape(m['handle'])} · {eng.RELATION_LABEL[m['relation']]}{tag}</span>
                <span><span class="pill" style="background:{LEVEL_COLOR[m['level']]}">{m['level']} {m['risk']:.0f}</span>
                &nbsp;{m['time']}</span></div>{html.escape(m['text'])}</div>""", unsafe_allow_html=True)

        st.markdown("#### 🕸️ Receiver's network")
        only_foll = np.setdiff1d(foll_r, mut_r)
        only_fing = np.setdiff1d(fing_r, mut_r)
        radial_fig(f"user_{r_uid}", BLUE,
                   [("Mutual", names_of(mut_r, 12), GREEN, 11),
                    ("Followers", names_of(only_foll, 14), BLUE, 9),
                    ("Following", names_of(only_fing, 14), PURPLE, 9)], key="left_net")

    # ------------------------------------------------ RIGHT: SENDER
    with right:
        mode = st.radio("Sender type", ["🤖 Fake account (create)", "👥 Genuine follower",
                                         "🤝 Mutual connection", "🌐 Genuine stranger"], horizontal=True)
        is_fake = mode.startswith("🤖")
        accent = RED if is_fake else GREEN
        title = "🟥 FAKE ACCOUNT · SENDER" if is_fake else "🟩 GENUINE ACCOUNT · SENDER"
        st.markdown(f'<div class="panel-title" style="background:{accent}">{title}</div>', unsafe_allow_html=True)

        if is_fake:
            with st.expander("⚙️ Configure the fake account", expanded=True):
                c1, c2 = st.columns(2)
                handle = c1.text_input("Username", value="free_prize_official")
                age_days = c2.slider("Account age (days)", 0, 1500, 12)
                c3, c4 = st.columns(2)
                n_follow = c3.slider("Following", 0, 3000, 600)
                n_foller = c4.slider("Followers", 0, 1000, 6)
                c5, c6 = st.columns(2)
                bot_mode = c5.checkbox("Bot behaviour (mass-follow, no follow-backs)", value=True)
                bot_net = c6.checkbox("Part of a bot network (follows/followed by other bots)", value=True)
                c7, c8 = st.columns(2)
                follows_me = c7.checkbox("Fake follows the receiver", value=False)
                i_follow_back = c8.checkbox("Receiver follows it back", value=False, disabled=not follows_me)
                mutual_n = st.slider("Follow-backs from followed accounts", 0, min(max(n_follow, 1), 300), 0,
                                     disabled=bot_mode)
                if st.button("🎲 Regenerate random connections"):
                    st.session_state.fake_seed += 1

            rng = np.random.default_rng(st.session_state.fake_seed * 1000 + n_follow * 7 + n_foller * 13
                                        + int(bot_mode) + 2 * int(bot_net))
            k = int(min(n_follow, eng.n - 1))
            if bot_mode:
                out_idx = rng.choice(eng.n, k, replace=False)
            else:
                pop = (eng.in_deg + 1) / (eng.in_deg + 1).sum()
                out_idx = rng.choice(eng.n, k, replace=False, p=pop)
            m = 0 if bot_mode else min(mutual_n, k)
            f_total = max(n_foller, m)
            in_idx = np.concatenate([out_idx[:m], rng.choice(eng.n, min(f_total - m, eng.n), replace=False)]) \
                if f_total else np.array([], dtype=int)
            if bot_net and len(eng.bot_idx) >= 4:
                out_idx = np.concatenate([out_idx, rng.choice(eng.bot_idx, 4, replace=False)])
                in_idx = np.concatenate([in_idx, rng.choice(eng.bot_idx, 4, replace=False)])
            if follows_me:
                out_idx = np.append(out_idx, r_idx)
                if i_follow_back:
                    in_idx = np.append(in_idx, r_idx)
            out_idx, in_idx = np.unique(out_idx).astype(int), np.unique(in_idx).astype(int)
            gnn_p, x = eng.score_new(out_idx, in_idx)
            s_follows_r, r_follows_s = bool(follows_me), bool(follows_me and i_follow_back)
            relation = ("mutual" if s_follows_r and r_follows_s else
                        "sender_follows_me" if s_follows_r else "none")
            sender = dict(uid=9_000_000_000 + st.session_state.fake_seed, handle=handle or "fake_account",
                          followers=len(in_idx), following=len(out_idx), age_days=age_days, relation=relation,
                          gnn_p=gnn_p, x=x, is_fake=True, out_idx=out_idx, in_idx=in_idx,
                          mutual_n=len(np.intersect1d(out_idx, in_idx)))
            badge, emoji = "UNVERIFIED", "🤖"
            bio = "Newly created account. You created this fake profile for the demo."
        else:
            rs = np.random.default_rng(int(r_idx) + 11)
            if mode.startswith("👥"):
                pool = np.setdiff1d(foll_r, mut_r)
                pool = pool if len(pool) else foll_r
                label = "Pick a genuine follower of the receiver"
            elif mode.startswith("🤝"):
                pool = mut_r
                label = "Pick a mutual connection"
            else:
                cand = rs.choice(eng.n, 400, replace=False)
                linked = set(foll_r.tolist()) | set(fing_r.tolist()) | {r_idx}
                pool = np.array([c for c in cand if c not in linked and eng.out_deg[c] > 3])
                label = "Pick a genuine stranger (no link to the receiver)"
            if len(pool) == 0:
                st.warning("No account of this type exists for this receiver - try another receiver ID.")
                st.stop()
            pool = rs.permutation(pool)[:20]
            sel = st.selectbox(label, [int(i) for i in pool], format_func=lambda i: f"@user_{eng.node_ids[i]}")
            sidx = int(sel)
            gnn_p, x = eng.score_existing(sidx)
            signup = estimate_signup(int(eng.node_ids[sidx]))
            days = (date.today() - signup).days
            relation = eng.relation(r_idx, sidx)
            sender = dict(uid=int(eng.node_ids[sidx]), handle=f"user_{eng.node_ids[sidx]}",
                          followers=int(eng.in_deg[sidx]), following=int(eng.out_deg[sidx]), age_days=days,
                          relation=relation, gnn_p=gnn_p, x=x, is_fake=False,
                          out_idx=eng.following(sidx), in_idx=eng.followers(sidx),
                          mutual_n=len(eng.mutuals(sidx)))
            badge, emoji = {"mutual": "MUTUAL", "sender_follows_me": "FOLLOWS YOU"}.get(relation, "NOT CONNECTED"), "🙂"
            bio = "Genuine account taken from the Twitter network dataset."

        st.session_state.cur_sender = sender
        st.markdown(profile_card(sender["handle"], sender["uid"], sender["followers"], sender["following"],
                                 "Mutual", f"{sender['mutual_n']:,}", age_text(sender["age_days"]),
                                 badge, accent, bio, emoji), unsafe_allow_html=True)
        st.caption(f"Relation to receiver: **{eng.RELATION_LABEL[sender['relation']]}**")

        st.markdown("#### ✉️ Compose message")
        st.selectbox("Quick message", list(PRESETS), index=1, key="preset", on_change=set_preset)
        st.text_area("Message to the receiver", key="msg", height=90)
        st.button("🚀 Send message to the original account", on_click=send_message,
                  use_container_width=True, type="primary")
        flash = st.session_state.flash
        if flash:
            {"sent": st.success, "blocked": st.error, "warn": st.warning}[flash[0]](flash[1])

        st.markdown("#### 🕸️ Sender's network")
        out_show = [i for i in sender["out_idx"] if i != r_idx]
        in_show = [i for i in sender["in_idx"] if i != r_idx]
        bots_o = [i for i in out_show if i >= eng.n]
        bots_i = [i for i in in_show if i >= eng.n]
        groups = [("Following", names_of([i for i in out_show if i < eng.n], 18), PURPLE, 8),
                  ("Followers", names_of([i for i in in_show if i < eng.n], 8), AMBER, 9)]
        if bots_o or bots_i:
            groups.append(("Bot network", names_of(list(bots_o) + list(bots_i), 8), RED, 12))
        if r_idx in sender["out_idx"] or r_idx in sender["in_idx"]:
            groups.append(("RECEIVER", [f"@user_{r_uid} (receiver)"], BLUE, 18))
        radial_fig(f"@{sender['handle']}", accent, groups, key="right_net")

        if st.session_state.last:
            st.markdown("---")
            show_analysis(st.session_state.last, key=len(st.session_state.inbox))

# ================================================================== PERFORMANCE
with tab_perf:
    M = eng.metrics
    st.subheader("Model comparison (30% held-out test nodes)")
    rows = [M[k] for k in ("gnn", "rf", "lr")]
    df = pd.DataFrame([{"Model": r["name"], "Accuracy": r["accuracy"], "Precision": r["precision"],
                        "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["auc"]} for r in rows])
    st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns[1:]}), use_container_width=True, hide_index=True)
    st.caption("Bots are synthetic (injected into the graph), so these numbers show how well the model "
               "separates that simulated bot behaviour from real users - not real-world Twitter spam.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for r, col in zip(rows, [RED, BLUE, AMBER]):
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{r['name']} (AUC {r['auc']:.3f})",
                                     line=dict(color=col, width=3 if r is rows[0] else 2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                 showlegend=False))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate",
                          height=380, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True, key="roc")
    with c2:
        cm = np.array(M["gnn"]["cm"])
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: Real", "Pred: Fake"], y=["True: Real", "True: Fake"],
                                   text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title="GNN confusion matrix", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, key="cm")
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(go.Scatter(y=M["loss_curve"], mode="lines+markers", line=dict(color=RED)))
        fig.update_layout(title="GNN training loss", xaxis_title="Epoch", yaxis_title="Loss", height=320)
        st.plotly_chart(fig, use_container_width=True, key="loss")
    with c4:
        imp = M["rf_importance"]
        fig = go.Figure(go.Bar(x=list(imp.values()), y=list(imp.keys()), orientation="h", marker_color=BLUE))
        fig.update_layout(title="Which graph features matter (Random Forest)", height=320,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, key="imp")

# ================================================================== ABOUT
with tab_about:
    st.subheader("Detection pipeline")
    st.markdown("""
1. **Twitter graph** - follower edges from `twitter_combined.txt`.
2. **Synthetic bots** - injected so the model has fake accounts to learn from (mass-follow, few followers,
   almost no follow-backs, bot rings).
3. **Node features** - following, followers, ratio, reciprocity, PageRank.
4. **GraphSAGE-style GNN** - every account is described by its own features *plus* the average features of
   the accounts it follows and its followers, over 2 hops.
5. **Live scoring** - when you send a message, the fake account's connections are added to the graph and the
   GNN scores it on the spot. Real accounts use their stored score.
6. **Text classifier** - TF-IDF + Logistic Regression on the message.
7. **Risk engine** - 50% GNN + 35% text + 15% account age, then a **capped** trust discount for
   followers / mutual connections. The discount shrinks when the GNN is suspicious, so a bot cannot become
   "safe" simply by following the receiver.
8. **Alert** - LOW (no alert), MEDIUM (caution), HIGH (red alert with Block / Report).
""")
    st.caption("Account age for dataset users is estimated from the user ID (older IDs = older accounts), "
               "because the dataset has no timestamps.")
