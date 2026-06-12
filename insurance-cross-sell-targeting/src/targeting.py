from __future__ import annotations

import numpy as np
import pandas as pd


def assign_campaign_priority(scores, top_target_pct: float = 0.20, hold_pct: float = 0.30):
    scores = np.asarray(scores)
    n = len(scores)
    order = np.argsort(-scores)

    labels = np.array(["low_priority"] * n, dtype=object)
    target_cut = max(1, int(n * top_target_pct))
    hold_cut = max(target_cut, int(n * (top_target_pct + hold_pct)))

    labels[order[:target_cut]] = "target"
    labels[order[target_cut:hold_cut]] = "hold"
    labels[order[hold_cut:]] = "low_priority"
    return labels


def build_lift_gain_table(y_true, scores, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "score": scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["bucket"] = pd.qcut(df.index + 1, q=n_bins, labels=False)

    total_positives = df["y"].sum()
    rows = []
    cumulative_pos = 0

    for b in sorted(df["bucket"].unique()):
        g = df[df["bucket"] == b]
        positives = g["y"].sum()
        cumulative_pos += positives
        capture_rate = cumulative_pos / total_positives if total_positives else 0.0

        rows.append(
            {
                "decile": int(b) + 1,
                "rows": int(len(g)),
                "positives": int(positives),
                "cumulative_positives": int(cumulative_pos),
                "capture_rate": float(capture_rate),
                "lift_vs_random": float(capture_rate / ((b + 1) / n_bins)) if (b + 1) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def generate_batch_target_list(base_df: pd.DataFrame, scores, id_col: str = "id", top_target_pct: float = 0.20, hold_pct: float = 0.30):
    out = base_df.copy().reset_index(drop=True)
    out["score"] = scores
    out["campaign_priority"] = assign_campaign_priority(scores, top_target_pct=top_target_pct, hold_pct=hold_pct)
    out = out.sort_values("score", ascending=False)
    cols = [c for c in [id_col, "score", "campaign_priority"] if c in out.columns]
    return out[cols]
