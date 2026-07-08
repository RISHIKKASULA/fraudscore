"""Generate the committed synthetic fixture: data/fixtures/synthetic.csv.

Same schema as the ULB credit-card dataset (Time, V1..V28, Amount, Class), ~2,000 rows,
inflated fraud rate (~5%) so tests and CI see enough positives. Seeded and deterministic:
running this script twice produces byte-identical output. CI runs the full pipeline on this
fixture; the real data never enters CI.

The V columns imitate PCA components: zero-mean for legit rows with decaying scale, and a
decaying, alternating-sign mean shift for fraud rows so models have real signal to find.
Amounts are lognormal with a heavier tail for fraud. `Time` spans ~2 days like the original.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000
FRAUD_RATE = 0.05
TIME_SPAN_S = 172_800  # ~2 days, like the original dataset
N_COMPONENTS = 28

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "synthetic.csv"


def make_fixture(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Labels: exact fraud count so the fixture's base rate is stable.
    n_fraud = round(n_rows * FRAUD_RATE)
    y = np.zeros(n_rows, dtype=np.int64)
    y[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    # Time: sorted seconds since first transaction (a sequence index, not clock time).
    time_s = np.sort(rng.uniform(0, TIME_SPAN_S, size=n_rows)).round(1)

    # V1..V28: PCA-like components with decaying scale; fraud rows get a decaying,
    # alternating-sign mean shift (strongest on the leading components, like the real data).
    scales = np.linspace(2.0, 0.3, N_COMPONENTS)
    shifts = np.array([(-1) ** j * 2.5 * scales[j] * np.exp(-j / 6) for j in range(N_COMPONENTS)])
    v = rng.normal(0.0, scales, size=(n_rows, N_COMPONENTS))
    v[y == 1] += shifts
    v = v.round(6)

    # Amount: lognormal; fraud has a heavier right tail.
    amount = np.where(
        y == 1,
        rng.lognormal(mean=4.0, sigma=1.4, size=n_rows),
        rng.lognormal(mean=3.2, sigma=1.2, size=n_rows),
    ).round(2)

    frame = pd.DataFrame(
        {
            "Time": time_s,
            **{f"V{j + 1}": v[:, j] for j in range(N_COMPONENTS)},
            "Amount": amount,
            "Class": y,
        }
    )
    return frame


def main() -> None:
    frame = make_fixture()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(frame)} rows, {int(frame['Class'].sum())} fraud)")


if __name__ == "__main__":
    main()
