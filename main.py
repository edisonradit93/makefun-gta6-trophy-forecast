"""Quantile forecaster for trophy earn rates."""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.5

TIER_DRIFT_WEIGHTS = {
    "bronze": 0.04,
    "silver": 0.02,
    "gold": 0.01,
    "platinum": 0.001,
}


def clamp(val: float) -> float:
    return min(99.99, max(0.01, val))


def to_logit(rate: float) -> float:
    c = clamp(rate)
    return math.log(c / (100.0 - c))


def from_logit(val: float) -> float:
    return 100.0 / (1.0 + math.exp(-val))


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    val = whole / 10000
    return int(val) if val.is_integer() else val


def predict_trophy_quantiles(
    trophy: dict,
    rates_history: list[float | None],
    horizon_days: float,
) -> list[float | int]:
    valid_rates = [r for r in rates_history if r is not None]
    if not valid_rates:
        return [0, 0, 0, 0, 0]

    last_rate = valid_rates[-1]
    if last_rate >= 99.99:
        return [100, 100, 100, 100, 100]

    y_last = to_logit(last_rate)

    # Estimate empirical trend if multiple observations exist
    drift = 0.0
    if len(valid_rates) >= 2:
        first_valid = valid_rates[0]
        dy = to_logit(last_rate) - to_logit(first_valid)
        drift = max(0.0, dy / len(valid_rates))

    tier = str(trophy.get("tier", "bronze")).lower()
    tier_drift = TIER_DRIFT_WEIGHTS.get(tier, 0.02)
    effective_drift = max(0.0, drift + tier_drift)

    spread = SIGMA * math.sqrt(horizon_days)

    quantiles: list[float | int] = []
    min_rate = last_rate

    for z in Z_SCORES:
        pred_logit = y_last + effective_drift * horizon_days + z * spread
        pred_pct = from_logit(pred_logit)
        # Monotonicity constraint: earn rate cannot drop below observed rate
        clamped_pred = max(min_rate, pred_pct)
        quantiles.append(round4(min(100.0, clamped_pred)))

    # Ensure strictly non-decreasing
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]

    return quantiles


def process_case(case_data: dict) -> dict:
    history = case_data.get("history", [])
    trophies = case_data.get("trophies", [])
    horizon_minutes = float(case_data.get("horizon_minutes", 90))
    horizon_days = horizon_minutes / 1440.0

    history_by_trophy: list[list[float | None]] = [[] for _ in trophies]
    for snap in history:
        rates = snap.get("rates", [])
        for i in range(len(trophies)):
            r = rates[i] if i < len(rates) else None
            history_by_trophy[i].append(r)

    rows = []
    for i, trophy in enumerate(trophies):
        t_id = trophy["id"]
        q_values = predict_trophy_quantiles(trophy, history_by_trophy[i], horizon_days)
        rows.append({"id": t_id, "q": q_values})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_file in sorted(INPUT_DIR.glob("*.in")):
        case_data = json.loads(in_file.read_text(encoding="utf-8"))
        forecast = process_case(case_data)
        out_file = OUTPUT_DIR / f"{in_file.stem}.out"
        out_file.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
