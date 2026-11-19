import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Coupon BTTS", layout="centered")

LIGUES = {"E0": "Angleterre D1", "E1": "Angleterre D2", "SC0": "Écosse D1",
          "D1": "Allemagne D1", "D2": "Allemagne D2", "I1": "Italie D1",
          "I2": "Italie D2", "SP1": "Espagne D1", "SP2": "Espagne D2",
          "F1": "France D1", "F2": "France D2", "N1": "Pays-Bas D1",
          "B1": "Belgique D1", "P1": "Portugal D1", "T1": "Turquie D1",
          "G1": "Grèce D1"}
SAISONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
        "HC", "AC", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
        "BbAvH", "BbAvD", "BbAvA", "BbAv>2.5", "BbAv<2.5"]
BASE = ["A_marque", "A_encaisse", "BTTS", "Marques", "Encaisses",
        "Tirs", "TirsContre", "Cadres", "CadresContre", "Corners"]


def probas_marche(t):
    t["p_over"] = (1 / t["cote_O25"]) / (1 / t["cote_O25"] + 1 / t["cote_U25"])
    s = 1 / t["cote_H"] + 1 / t["cote_N"] + 1 / t["cote_A"]
    t["p_H"] = (1 / t["cote_H"]) / s
    t["p_N"] = (1 / t["cote_N"]) / s
    t["p_A"] = (1 / t["cote_A"]) / s
    t["ecart_force"] = (t["p_H"] - t["p_A"]).abs()
    return t


@st.cache_data(ttl=6 * 3600, show_spinner="Analyse des matchs en cours...")
def calculer():
    tous = []
    for saison in SAISONS:
        for code, nom in LIGUES.items():
            try:
                d = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{saison}/{code}.csv")
                d = d.reindex(columns=COLS)
                d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
                d["Ligue"] = nom
                tous.append(d)
            except Exception:
                pass
    df = pd.concat(tous, ignore_index=True)
    df["cote_H"] = df["AvgH"].fillna(df["BbAvH"])
    df["cote_N"] = df["AvgD"].fillna(df["BbAvD"])
    df["cote_A"] = df["AvgA"].fillna(df["BbAvA"])
    df["cote_O25"] = df["Avg>2.5"].fillna(df["BbAv>2.5"])
    df["cote_U25"] = df["Avg<2.5"].fillna(df["BbAv<2.5"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["idx"] = df.index
    df = probas_marche(df)

    def vue(lieu, e, m, en, t, tc, c, cc, co):
        return pd.DataFrame({"idx": df["idx"], "lieu": lieu, "Date": df["Date"],
                             "Equipe": df[e], "Marques": df[m], "Encaisses": df[en],
                             "Tirs": df[t], "TirsContre": df[tc], "Cadres": df[c],
                             "CadresContre": df[cc], "Corners": df[co], "BTTS": df["BTTS"]})

    eq = pd.concat([vue("dom", "HomeTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC"),
                    vue("ext", "AwayTeam", "FTAG", "FTHG", "AS", "HS", "AST", "HST", "AC")]
                   ).sort_values(["Equipe", "Date", "idx"]).reset_index(drop=True)
    eq["A_marque"] = (eq["Marques"] > 0).astype(int)
    eq["A_encaisse"] = (eq["Encaisses"] > 0).astype(int)
    g = eq.groupby("Equipe")
    for c in BASE:
        eq["f_" + c] = g[c].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    fc = ["f_" + c for c in BASE]
    dom_f = eq[eq["lieu"] == "dom"].set_index("idx")[fc].add_prefix("dom_")
    ext_f = eq[eq["lieu"] == "ext"].set_index("idx")[fc].add_prefix("ext_")
    data = df.set_index("idx").join(dom_f).join(ext_f).reset_index()
    data = data.sort_values("Date").reset_index(drop=True)
    data["f_ligue"] = data.groupby("Ligue")["BTTS"].transform(
        lambda s: s.shift(1).expanding(min_periods=50).mean())
    stats = ["f_ligue"] + list(dom_f.columns) + list(ext_f.columns)
    marche = ["p_over", "p_H", "p_N", "p_A", "ecart_force"]
    cols = stats + marche
    data = data.dropna(subset=cols)

    sc = StandardScaler()
    modele = LogisticRegression(max_iter=2000).fit(sc.fit_transform(data[cols]), data["BTTS"])

    try:
        fx = pd.read_csv("https://www.football-data.co.uk/fixtures.csv")
    except Exception:
        fx = pd.read_csv("https://www.football-data.co.uk/fixtures.csv", encoding="latin-1")
    fx["Ligue"] = fx["Div"].map(LIGUES)
    fx = fx.dropna(subset=["Ligue", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")

    def cote(a, b):
        s = fx[a] if a in fx.columns else pd.Series(np.nan, index=fx.index)
        return s.fillna(fx[b]) if b in fx.columns else s

    fx["cote_H"], fx["cote_N"], fx["cote_A"] = cote("AvgH", "B365H"), cote("AvgD", "B365D"), cote("AvgA", "B365A")
    fx["cote_O25"], fx["cote_U25"] = cote("Avg>2.5", "B365>2.5"), cote("Avg<2.5", "B365<2.5")
    fx = probas_marche(fx)
    last = eq.groupby("Equipe").tail(10).groupby("Equipe")[BASE].mean()
    last.columns = ["f_" + c for c in last.columns]
    dom = last.add_prefix("dom_").reindex(fx["HomeTeam"]).reset_index(drop=True)
    ext = last.add_prefix("ext_").reindex(fx["AwayTeam"]).reset_index(drop=True)
    fx = pd.concat([fx, dom, ext], axis=1)
    fx["f_ligue"] = fx["Ligue"].map(df.groupby("Ligue")["BTTS"].mean())
    ok = fx.dropna(subset=cols).copy()
    ok["proba"] = modele.predict_proba(sc.transform(ok[cols]))[:, 1]
    return ok


st.title("⚽ Coupon BTTS du jour")
ok = calculer()
if ok.empty:
    st.warning("Aucun match analysable pour le moment.")
    st.stop()

with st.sidebar:
    st.header("Réglages")
    mise = st.number_input("Mise (francs)", value=20000, step=1000)
    cote_min = st.number_input("Cote min par match", value=1.30, step=0.05)
    cote_max = st.number_input("Cote max par match", value=1.90, step=0.05)
    total_min = st.number_input("Cote totale min", value=3.3, step=0.1)
    total_max = st.number_input("Cote totale max", value=3.7, step=0.1)
    nb_matchs = st.slider("Matchs à saisir", 4, 12, 7)

dates = sorted(ok["Date"].dropna().dt.date.unique())
aujourdhui = pd.Timestamp.today().date()
defaut = dates.index(aujourdhui) if aujourdhui in dates else 0
jour = st.selectbox("Jour", dates, index=defaut)

top = ok[ok["Date"].dt.date == jour].nlargest(nb_matchs, "proba")
tab = pd.DataFrame({
    "Match": top["HomeTeam"].values + " - " + top["AwayTeam"].values,
    "Ligue": top["Ligue"].values,
    "Proba %": (top["proba"].values * 100).round(1),
    "Cote juste": (1 / top["proba"].values).round(2),
    "Ta cote BTTS Oui": np.nan,
})
st.subheader("Matchs les plus probables")
st.caption("Tape la cote BTTS « Oui » de ton bookmaker dans la dernière colonne.")
edite = st.data_editor(tab, hide_index=True, use_container_width=True,
                       disabled=["Match", "Ligue", "Proba %", "Cote juste"])

m = edite.dropna(subset=["Ta cote BTTS Oui"])
m = m[(m["Ta cote BTTS Oui"] >= cote_min) & (m["Ta cote BTTS Oui"] <= cote_max)].reset_index(drop=True)

if len(m) < 2:
    st.info("Saisis au moins 2 cotes (entre %.2f et %.2f) pour voir les coupons." % (cote_min, cote_max))
else:
    res = []
    for n in (2, 3):
        for t in combinations(range(len(m)), n):
            s = m.iloc[list(t)]
            cc = s["Ta cote BTTS Oui"].prod()
            if total_min <= cc <= total_max:
                res.append({"Matchs": " | ".join(s["Match"]), "Nb": n,
                            "Cote combinée": round(cc, 2),
                            "Proba de gain %": round((s["Proba %"] / 100).prod() * 100, 1),
                            "Gain si ça passe": int(mise * (cc - 1))})
    st.subheader("Meilleurs coupons")
    if res:
        st.dataframe(pd.DataFrame(res).sort_values("Proba de gain %", ascending=False).head(5),
                     hide_index=True, use_container_width=True)
    else:
        st.warning("Aucun coupon avec une cote totale entre %.1f et %.1f. Ajoute des matchs ou change les réglages." % (total_min, total_max))

st.caption("Outil d'aide à la décision, sans garantie. Un coupon de ce type gagne environ 1 jour sur 4 à 5. Ne mise que ce que tu peux perdre.")
