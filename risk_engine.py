"""
============================================================
 risk_engine.py  -  Combines GNN score + message text + account age
============================================================
Final risk (0-100) =
      50% x GNN fake-account probability   (graph behaviour)
    + 35% x message spam probability       (text classifier)
    + 15% x account-age risk
  then reduced by a CAPPED trust discount if the sender follows / is
  followed by the receiver. The discount shrinks as the GNN probability
  rises, so a bot cannot become "safe" just by following the receiver.
"""
import os
from datetime import date

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ------------------------------------------------------------------ account age
# SNAP data has no timestamps; Twitter IDs grew over time, so ID -> approx. sign-up date.
_ANCHORS = [(12, 2006.22), (428_333, 2006.95), (783_214, 2007.12), (813_286, 2007.20),
            (11_348_282, 2007.95), (14_230_524, 2008.20), (15_439_395, 2008.55),
            (17_919_972, 2008.95), (19_058_681, 2009.05), (21_447_363, 2009.13),
            (25_365_536, 2009.20), (27_260_086, 2009.22), (44_196_397, 2009.45),
            (79_293_791, 2009.80), (150_000_000, 2010.30), (300_000_000, 2011.40),
            (500_000_000, 2012.10), (1_000_000_000, 2012.95)]
_IDS = np.array([a[0] for a in _ANCHORS], dtype=float)
_YRS = np.array([a[1] for a in _ANCHORS], dtype=float)


def estimate_signup(user_id):
    yr = float(np.interp(user_id, _IDS, _YRS))
    year = int(yr)
    return date.fromordinal(date(year, 1, 1).toordinal() + int((yr - year) * 365))


def age_text(days):
    years, rem = divmod(int(days), 365)
    months = min(rem // 30, 11)
    if years == 0:
        return f"{int(days)} days" if days < 60 else f"{months} months"
    return f"{years}y {months}m"


def age_risk(days):
    return float(np.interp(days, [0, 30, 180, 365, 1095], [1.0, 0.8, 0.45, 0.2, 0.0]))


# ------------------------------------------------------------------ text model
SPAM_TRAIN = [
    "Congratulations! You have won a free cash prize. Click the link to claim now!",
    "URGENT: your account will be suspended. Verify your password immediately",
    "Claim your free iPhone today, limited offer, click here",
    "You won $1000 reward! Send your bank details to receive the cash",
    "Earn money fast from home, no experience needed, join now",
    "Limited time bonus! Click this link to get 50% off, act now",
    "Free gift card waiting for you, claim before it expires",
    "Your package is held, pay the small fee at this link to release it",
    "Win a brand new car! Reply YES to enter the lucky draw",
    "Make $500 a day with this secret trick, click now",
    "Dear user, you have been selected for an exclusive prize, claim today",
    "Get free followers and likes instantly, click the link in bio",
    "Hot singles near you are waiting, click to chat now",
    "Your bank account is locked. Login here to restore access urgently",
    "Exclusive crypto giveaway! Double your bitcoin, send now",
    "Cheap loans approved instantly, no credit check, apply now",
    "Final notice: you owe taxes, pay immediately to avoid arrest",
    "Lowest price pills online, order now, free shipping",
    "Click here to unlock your reward, offer ends tonight",
    "You are our lucky winner! Claim your free vacation now",
    "FREE entry in a weekly competition to win cash, text WIN to 80085",
    "Congratulations, your number was picked for a free prize, call now",
    "Verify your account now or it will be deleted, click the link",
    "Get rich quick with our investment plan, guaranteed returns",
    "Free money! Claim your bonus before midnight, click here",
    "Special promotion just for you, buy one get ten free, hurry",
    "Your Netflix payment failed, update billing details at this link",
    "Work from home and earn thousands weekly, sign up free",
    "Discount watches and bags, huge sale, click to shop now",
    "Urgent reply needed: claim your inheritance of 5 million dollars",
    "Dear winner, send your OTP to confirm your cash reward",
    "Increase your followers 10000 for free, limited slots",
    "Download this free app and win real cash prizes daily",
    "Act now! This limited offer will expire in one hour, click to claim",
]
HAM_TRAIN = [
    "Hey, are you free for a call tomorrow?",
    "Can we move the meeting to 3 pm?",
    "Thanks for sharing the notes, they were really helpful",
    "Happy birthday! Hope you have a wonderful day",
    "I will be late, stuck in traffic, start without me",
    "Did you finish the assignment for the GNN project?",
    "Let's meet for coffee this weekend",
    "Great tweet today, I totally agree with your point",
    "What time does the movie start tonight?",
    "Please send me the report when you get a chance",
    "Congrats on your new job, well deserved!",
    "Are you coming to the college fest on Saturday?",
    "I read your article, it was very interesting",
    "Can you help me debug this code later?",
    "Dinner at my place tonight, bring your friends",
    "Just reached home, will call you in the morning",
    "Good luck for your exam tomorrow, you will do great",
    "Did you see the match yesterday? What a game",
    "Thanks for the follow, love your content",
    "Sorry I missed your call, I was in a lecture",
    "Can you share the dataset link for the mini project?",
    "The presentation went well, thank you for your help",
    "Remind me to buy groceries on the way back",
    "I am working on the Streamlit dashboard right now",
    "How was your trip? Send some pictures when you can",
    "Let me know if you need anything from the market",
    "We should collaborate on the research paper",
    "Wishing you and your family a happy festival",
    "I liked your latest post about machine learning",
    "Mom asked if you are coming home this weekend",
    "Which laptop are you planning to buy?",
    "See you at the library after class",
    "Could you review my pull request when you are free?",
    "The weather is great today, want to go for a walk?",
    "Thank you so much for the recommendation",
    "Please find the attached document for tomorrow's meeting",
]
SPAM_WORDS = ["win", "won", "prize", "reward", "free", "claim", "cash", "urgent", "bonus", "click",
              "limited", "offer", "verify", "password", "bitcoin", "giveaway", "winner", "otp"]


class TextSpamModel:
    """TF-IDF + Logistic Regression. Drop the UCI SMS Spam file (spam.csv,
    columns v1/v2) next to this script and it is used automatically too."""

    def __init__(self):
        texts = SPAM_TRAIN + HAM_TRAIN
        labels = [1] * len(SPAM_TRAIN) + [0] * len(HAM_TRAIN)
        csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spam.csv")
        if os.path.exists(csv):
            try:
                import pandas as pd
                df = pd.read_csv(csv, encoding="latin-1")[["v1", "v2"]].dropna()
                texts += df["v2"].astype(str).tolist()
                labels += (df["v1"] == "spam").astype(int).tolist()
            except Exception:
                pass
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
            LogisticRegression(C=8, max_iter=1000))
        self.model.fit(texts, labels)

    def prob(self, msg):
        return float(self.model.predict_proba([msg])[0, 1])


# ------------------------------------------------------------------ engine
class RiskEngine:
    W_GNN, W_TEXT, W_AGE = 0.50, 0.35, 0.15
    TRUST = {"mutual": 0.35, "sender_follows_me": 0.18, "i_follow_sender": 0.08, "none": 0.0}
    RELATION_LABEL = {"mutual": "Mutual (follow each other)", "sender_follows_me": "Follows you",
                      "i_follow_sender": "You follow them", "none": "No direct link"}

    def __init__(self, bundle_path):
        b = joblib.load(bundle_path)
        self.node_ids = b["node_ids"]
        self.n = len(self.node_ids)
        self.csr = b["A"].tocsr()
        self.csc = self.csr.tocsc()
        self.X, self.H1 = b["X"], b["H1"]
        self.pr, self.outdeg_all = b["pr"], b["outdeg"]
        self.n_total = self.X.shape[0]
        self.scaler, self.gnn = b["scaler"], b["gnn"]
        self.mu, self.sd = b["mu"], b["sd"]
        self.gnn_prob = b["gnn_prob"]
        self.metrics, self.meta = b["metrics"], b["meta"]
        self.out_deg = np.diff(self.csr.indptr)
        self.in_deg = np.diff(self.csc.indptr)
        self.bot_idx = np.arange(self.n, self.n_total)   # synthetic bot nodes (for the demo's bot ring)
        self.text = TextSpamModel()

    # ---- graph lookups
    def lookup(self, uid):
        i = int(np.searchsorted(self.node_ids, uid))
        return i if i < self.n and self.node_ids[i] == uid else None

    def followers(self, i):
        return self.csc.indices[self.csc.indptr[i]:self.csc.indptr[i + 1]]

    def following(self, i):
        return self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i + 1]]

    def mutuals(self, i):
        return np.intersect1d(self.followers(i), self.following(i))

    def relation(self, receiver, sender):
        s_follows_r = bool(np.any(self.following(sender) == receiver))
        r_follows_s = bool(np.any(self.following(receiver) == sender))
        if s_follows_r and r_follows_s:
            return "mutual"
        if s_follows_r:
            return "sender_follows_me"
        if r_follows_s:
            return "i_follow_sender"
        return "none"

    def pick_default_receiver(self):
        cand = np.where((self.in_deg >= 30) & (self.in_deg <= 400) & (self.out_deg >= 20))[0]
        if len(cand) == 0:
            cand = np.argsort(-self.in_deg)[:50]
        cand = cand[np.argsort(-self.in_deg[cand])][:150]
        best = max(cand, key=lambda i: len(self.mutuals(i)))
        return int(best)

    # ---- GNN scoring
    def score_existing(self, i):
        return float(self.gnn_prob[i]), self.X[i]

    def score_new(self, out_idx, in_idx):
        """Score a brand-new account that follows `out_idx` and is followed by `in_idx`."""
        out_idx = np.unique(np.asarray(out_idx, dtype=int))
        in_idx = np.unique(np.asarray(in_idx, dtype=int))
        n_out, n_in = len(out_idx), len(in_idx)
        mutual = len(np.intersect1d(out_idx, in_idx))
        recip = mutual / max(n_out, 1)
        s = float(np.sum(self.pr[in_idx] / np.maximum(self.outdeg_all[in_idx], 1))) if n_in else 0.0
        prf = np.log10(0.15 + 0.85 * self.n_total * s)
        x = np.array([np.log1p(n_out), np.log1p(n_in), np.log1p(n_out) - np.log1p(n_in),
                      recip, prf], dtype=np.float32)

        def mean(M, idx):
            return M[idx].mean(0) if len(idx) else np.zeros(M.shape[1], dtype=np.float32)

        h1 = np.concatenate([x, mean(self.X, out_idx), mean(self.X, in_idx)])
        h2 = np.concatenate([h1, mean(self.H1, out_idx), mean(self.H1, in_idx)])
        p = float(self.gnn.predict_proba(self.scaler.transform(h2[None, :]))[0, 1])
        return p, x

    # ---- text
    def text_prob(self, msg):
        return self.text.prob(msg)

    @staticmethod
    def keyword_hits(msg):
        low = msg.lower()
        return [w for w in SPAM_WORDS if w in low]

    # ---- explanation of the graph signal
    def explain(self, x):
        """Compare the account's own features with the normal user population."""
        z = (x - self.mu) / self.sd
        sign = np.array([1, -1, 1, -1, -1])          # direction that looks suspicious
        sus = z * sign
        vals = [f"{np.expm1(x[0]):,.0f}", f"{np.expm1(x[1]):,.0f}",
                f"{np.expm1(x[0]) / max(np.expm1(x[1]), 1):,.1f}x", f"{x[3] * 100:.1f}%",
                f"{10 ** x[4]:.2f}x avg"]
        labels = ["Accounts followed", "Followers", "Following / followers ratio",
                  "Follow-back rate (reciprocity)", "Network influence (PageRank)"]
        return [{"label": l, "value": v, "z": float(zz), "suspicion": float(s),
                 "flagged": bool(s > 1.5)} for l, v, zz, s in zip(labels, vals, z, sus)]

    # ---- final score
    def combine(self, gnn_p, text_p, age_days, relation, hits):
        a = age_risk(age_days)
        g_pts = 100 * self.W_GNN * gnn_p
        t_pts = 100 * self.W_TEXT * text_p
        a_pts = 100 * self.W_AGE * a
        base = g_pts + t_pts + a_pts
        discount = base * self.TRUST[relation] * (1 - gnn_p) ** 2   # shrinks fast as GNN prob rises
        risk = base - discount
        override = 0.0
        if gnn_p >= 0.80 and risk < 65:                         # strong structural evidence
            override = 65 - risk
        elif gnn_p >= 0.65 and risk < 60:                       # fairly strong structural evidence
            override = 60 - risk
        elif text_p >= 0.80 and gnn_p >= 0.40 and risk < 62:    # spam text + suspicious account
            override = 62 - risk
        elif text_p >= 0.80 and risk < 45:                      # clearly spammy text
            override = 45 - risk
        risk += override
        level = "HIGH" if risk >= 60 else ("MEDIUM" if risk >= 30 else "LOW")

        reasons = []
        if gnn_p >= 0.5:
            reasons.append(f"GNN: account's graph behaviour matches the fake/bot pattern ({gnn_p:.0%})")
        else:
            reasons.append(f"GNN: account's graph behaviour looks normal (fake probability {gnn_p:.0%})")
        if text_p >= 0.5:
            reasons.append(f"Text: message reads like spam ({text_p:.0%})"
                           + (f" - keywords: {', '.join(hits)}" if hits else ""))
        else:
            reasons.append(f"Text: message looks normal (spam probability {text_p:.0%})")
        if age_days < 30:
            reasons.append(f"Age: brand-new account ({int(age_days)} days old)")
        elif age_days < 180:
            reasons.append(f"Age: recently created account ({age_text(age_days)})")
        else:
            reasons.append(f"Age: established account ({age_text(age_days)})")
        if relation != "none":
            reasons.append(f"Trust: {self.RELATION_LABEL[relation]} - discount "
                           f"{discount:.1f} pts" + (" (reduced because GNN is suspicious)"
                                                    if gnn_p > 0.5 else ""))
        else:
            reasons.append("Trust: no direct follow link with the receiver")
        if override > 0:
            reasons.append("Override: strong single signal raised the final score")
        return {"risk": float(min(risk, 100)), "level": level, "gnn_p": gnn_p, "text_p": text_p,
                "age_days": age_days, "relation": relation, "hits": hits, "reasons": reasons,
                "parts": {"gnn": g_pts, "text": t_pts, "age": a_pts,
                          "discount": discount, "override": override}}
