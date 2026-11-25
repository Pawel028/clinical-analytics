# Clinic Operations Analytics

Interactive Dash dashboard for exploring clinic patient volume, waiting time and care scores,
with a quantified link between waiting and patient experience.

**Live demo:** _add your URL after deploying_

## What it does
- **Heatmap** of patient volume by weekday and hour; click a cell to drill into that hour.
- **KPI row and key findings** that update with the clinic, date range and admit source you select.
- **Wait-time target:** set a target (30 to 150 min) and see the share of visits that exceed it and
  how their care scores compare.
- **Wait vs experience chart:** average care score by wait-time band.
- **Department table** of individual waits and scores.
- **Model check:** a gradient-boosted model could not predict waiting time from time, place or admit
  source (test R-squared about 0), so the dashboard says so rather than showing a forecast
  (`scripts/model_check.py`).

## Credit and licence
Adapted from Plotly's open-source
[dash-clinical-analytics](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-clinical-analytics)
sample app (MIT). Original data and layout are Plotly's; the dataset is a demonstration dataset and
should be treated as illustrative, not real clinical data.

**My changes:** KPI tiles, auto-generated findings, wait-target analysis, wait-vs-experience chart,
offline model check, vectorised data pipeline and heatmap (faster start-up), inclusive end date,
updated for current Dash 4 / pandas 3, Docker and Render config, rebranded UI.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:8050
```
Set `PORTFOLIO_URL` to show a "Back to portfolio" link in the header.

## Deploy
Render: push this folder to GitHub, create a Web Service from the repo (it reads `render.yaml`
and the `Dockerfile`). Free instances sleep after 15 idle minutes and take about a minute to wake.
