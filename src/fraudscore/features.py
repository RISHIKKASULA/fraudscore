"""Feature pipeline: V1..V28 passthrough, Amount log1p+robust-scale, Time -> cycle_phase.

`cycle_phase` is phase within a 24-hour cycle relative to the dataset's first
transaction — it can capture daily periodicity but is NOT time-of-day.
"""
