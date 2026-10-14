"""
Yöntem çiftleri için görsel bazında eşleştirilmiş istatistik testleri (evaluate.py'den sonra çalıştırılır).

Girdi : results/per_image.csv
Çıktı : results/stats_pairwise.csv — her (veri seti, seviye, metrik, A, B) için:
          ort_fark  : A − B (işaret, "pozitif = A daha iyi" olacak şekilde ayarlı; MAE/RMSE'de düşük iyi)
          A_kazanir%: A'nın daha iyi olduğu görsel oranı,  B_kazanir%: tersi
          p         : sürekli metriklerde Wilcoxon işaretli sıra testi,
                      Top1-uyum (0/1) için kesin McNemar testi (uyuşmayan çiftlerde binom testi)
          p_holm    : tüm testler üzerinden Holm düzeltmeli p
Kullanım: python stats.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
METHODS = ["SMF", "AMF", "MDBUTMF", "EMPR", "HDMR", "DAMF", "DnCNN", "DnCNN-SP", "SeConvUNet"]
DL = {"DnCNN", "DnCNN-SP", "SeConvUNet"}
METRICS = {"PSNR(dB)": 1, "SSIM": 1, "MAE": -1, "RMSE": -1, "F1": 1, "Top1-uyum": 1}
ALPHA = 0.05


def paired_test(a, b, binary):
    if binary:
        n_a, n_b = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
        return binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    d = a - b
    return wilcoxon(d).pvalue if np.any(d != 0) else 1.0


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    d = pd.read_csv(os.path.join(RES, "per_image.csv"), dtype={"img": str})
    rows = []
    for (ds, lv), g in d.groupby(["Dataset", "Seviye"]):
        for met, sign in METRICS.items():
            w = g.pivot(index="img", columns="Yöntem", values=met)
            if w.isna().all().all():
                continue
            if w[METHODS].isna().any().any():
                raise ValueError(f"{ds} %{lv} {met}: eksik değer var")
            for i, A in enumerate(METHODS):
                for B in METHODS[i + 1:]:
                    a, b = w[A].to_numpy(), w[B].to_numpy()
                    diff = sign * (a - b)
                    rows.append({"Dataset": ds, "Seviye": lv, "Metrik": met, "A": A, "B": B, "n": len(diff),
                                 "ort_fark": diff.mean(),
                                 "A_kazanir%": 100 * np.mean(diff > 0), "B_kazanir%": 100 * np.mean(diff < 0),
                                 "p": paired_test(a, b, met == "Top1-uyum")})
    st = pd.DataFrame(rows)
    st["p_holm"] = holm(st["p"].to_numpy())
    st.to_csv(os.path.join(RES, "stats_pairwise.csv"), index=False)
    print(f"{len(st)} test → results/stats_pairwise.csv")

    sig = st[st["p_holm"] < ALPHA]
    nondl_wins = sig[sig["A"].isin(set(METHODS) - DL) & sig["B"].isin(DL) & (sig["ort_fark"] > 0)]
    print(f"\nDL olmayan bir yöntemin bir DL modelini anlamlı biçimde geçtiği durumlar (Holm p<{ALPHA}): {len(nondl_wins)}")
    for _, r in nondl_wins.iterrows():
        print(f"  {r.Dataset:13s} %{r.Seviye:2d} {r.Metrik:9s} {r.A:8s} > {r.B:10s}  fark={r.ort_fark:+.4f}"
              f"  {r.A} kazanır %{r['A_kazanir%']:.1f}  p_holm={r.p_holm:.1e}")

    print("\nDL olmayan yöntemler arasında en iyi ve ikinciye karşı anlamlılığı:")
    nondl = [m for m in METHODS if m not in DL]
    for (ds, lv), g in d[d["Yöntem"].isin(nondl)].groupby(["Dataset", "Seviye"]):
        parts = []
        for met, sign in METRICS.items():
            mean = g.groupby("Yöntem")[met].mean()
            if mean.isna().all():
                continue
            ranked = (sign * mean).sort_values(ascending=False).index
            b1, b2 = ranked[0], ranked[1]
            r = st[(st.Dataset == ds) & (st.Seviye == lv) & (st.Metrik == met) &
                   (((st.A == b1) & (st.B == b2)) | ((st.A == b2) & (st.B == b1)))].iloc[0]
            parts.append(f"{met}: {b1}{'*' if r.p_holm < ALPHA else ''} (2.: {b2})")
        print(f"  {ds:13s} %{lv:2d}  " + " | ".join(parts))
    print("  (* = ikinciden anlamlı biçimde iyi, Holm p<0.05)")


if __name__ == "__main__":
    main()
