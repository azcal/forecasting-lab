# MustBeMoose Forecasting Lab

Eight automated sports-forecasting pipelines (WNBA, MLB first-5, CS2, Big-5 soccer DNB,
NFL, NCAAF, NHL, NBA) built on public data, frozen historical training windows, and
untouched holdout seasons evaluated once. Live pipelines grade the full slate daily on
log loss and Brier score against base-rate and Elo floors via GitHub Actions.

Frozen-holdout log loss (model vs Elo floor): Soccer 0.575/0.590 · NCAAF 0.545/0.564
(also beats CFBD's published Elo) · NBA 0.590/0.616 · NFL 0.622/0.645 · CS2 0.622/0.623
· WNBA 0.633/0.655 · NHL 0.673/0.678 · MLB F5 0.691/0.692.

Reading those pairs matters as much as the left-hand number. Where the two are close, the
feature layer is contributing almost nothing and the rating is doing the work. CS2 is the
clearest case: its head is a monotone function of the Elo gap and adds +0.0010 log loss
over the raw rating (t = 0.37). The July 2026 rebuild therefore went after the rating
rather than the head, switching Elo updates from map counts to per-map round margin. That
moved the holdout from 0.6255 to 0.6220 and AUC from 0.6924 to 0.6980.

## Retired and never-shipped targets

Three targets were built, evaluated once on a frozen holdout, and shut off against a
stated bar. They are published on the dashboard alongside the live boards.

- **CS2 series length (O/U 2.5 maps)** — retired 2026-07-26. Beat its base rate by
  +0.0009 log loss on n = 623 (t = 0.27, p = 0.39), AUC 0.5241. Predictions spanned
  0.231 to 0.469 against a 0.4334 base rate, so the model could never call a long series.
  A resolution failure, which no amount of recalibration repairs.
- **NHL totals** — failed validation, never shipped. Did not clear its base rate without a
  shot-quality input the pipeline does not carry.
- **NCAAF lineup layer** — failed validation, never shipped. Player-availability features
  did not improve on the team-state model out-of-sample.

## Boards under review

**MLB F5** clears its base rate by roughly +0.0006 log loss across 1,300 live forecasts
(t = 0.13, p = 0.45). Its Elo floor sits above the base rate, meaning team strength carries
no information about who leads after five innings. A single-parameter temperature
calibration was added 2026-07-26 after the raw head measured overconfident by a factor of
about 1.7 on two out-of-sample splits. A retirement bar was pre-registered the same day and
is evaluated on every pipeline run: at the end of the 2026 regular season the calibrated
edge over base rate must reach +0.010 with p < 0.01 on the full live sample, or the board
is retired and written up.

## Method

Each pipeline builds strictly pregame features through a sequential state engine with no
look-ahead, trains on a frozen window, and treats later seasons as holdouts opened once.
Every live forecast on the full slate is graded, not a filtered subset. Rolling skill
monitors bench any board that drops below its base rate. Early-season forecasts publish
from day one but withhold actionability guidance until each league passes a games-played
threshold.

Model quality is judged on log loss and Brier score against frozen holdouts. Market-derived
metrics are deliberately excluded, since none of these models take market inputs.

Live app: https://mustbemoose-forecasting-lab.streamlit.app/
