# MustBeMoose Forecasting Lab

Eight automated sports-forecasting pipelines (WNBA, MLB first-5, CS2, Big-5 soccer DNB,
NFL, NCAAF, NHL, NBA) built on public data, frozen historical training windows, and
untouched holdout seasons evaluated once. Live pipelines grade the full slate daily on
log loss and Brier score against base-rate and Elo floors via GitHub Actions.

Frozen-holdout log loss (model vs Elo floor): Soccer 0.575/0.590 · NCAAF 0.545/0.564
(also beats CFBD's published Elo) · NBA 0.590/0.616 · NFL 0.622/0.645 · CS2 0.626/0.626
· WNBA 0.633/0.655 · NHL 0.673/0.678 · MLB F5 0.691/0.692. Two candidate targets failed
validation and never shipped; one live target is flagged and paused by the skill monitor.

Live app: https://mustbemoose-forecasting-lab.streamlit.app/ 
