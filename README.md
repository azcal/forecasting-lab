# MustBeMoose Forecasting Lab

Four automated sports-forecasting pipelines with a live model-monitoring dashboard.
Built end to end: data engineering, feature pipelines, probabilistic models,
walk-forward validation, CI/CD deployment, and automated model-health monitoring.

**Dashboard:** (Streamlit Cloud link goes here after deploy)

## What this demonstrates

- **Data engineering.** Ingestion from four public sources (MLB Stats API, ESPN/wehoop
  mirrors, bo3.gg, football-data.co.uk): ~35k historical events, 67k pitcher game logs,
  49k player-map rows, refreshed daily by scheduled jobs.
- **Feature engineering without leakage.** Sequential state engines compute every
  feature strictly pregame (exponentially decayed, opponent-adjusted ratings; player
  availability; schedule and fatigue effects). Features are recorded before results
  update state, by construction.
- **Honest evaluation.** Hyperparameters tuned on a single validation season. Later
  seasons are frozen holdouts evaluated exactly once. Every model must beat two floors
  out of sample: the base-rate constant and an Elo baseline. Two candidate models
  failed the floor and were rejected; one feature layer failed its single holdout test
  and was shelved. Those results are reported, not hidden.
- **Production automation.** GitHub Actions runs each pipeline daily: refresh data,
  forecast the full slate, grade yesterday, commit logs. Zero manual steps.
- **Model monitoring.** Rolling log-loss skill vs baseline with automated alerts.
  One forecast target is currently paused by its own alert, which is the system
  working as designed.

## Out-of-sample results (frozen holdouts)

| pipeline | target | holdout | log loss | baseline (base rate) | hit rate |
|---|---|---|---|---|---|
| Soccer (Big 5) | match winner, draws excluded | 2025-26, n=1,306 | **0.5753** | 0.6773 | 69.8% |
| CS2 (tier S/A) | series winner | 2026 YTD, n=623 | **0.6255** | 0.6853 | 64.4% |
| WNBA | game winner | 2026 YTD, n=202 | **0.6331** | 0.6955 | 62.9% |
| MLB | first-5-innings winner | 2025, n=2,039 | **0.6792** | 0.6899 | 56.1% |

## Stack

Python (pandas, NumPy, scikit-learn), GitHub Actions, Streamlit, Plotly.
All data sources and infrastructure are free.
