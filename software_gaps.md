\# Software Gap List — Weeks 2–4



\## Week 2: Real-time Backend

\- \[ ] sensor\_mapping.py — raw sensor unit conversions (C↔K, RPM calibration)

\- \[ ] simulate\_stream.py — streams CSV rows to /predict every 2-3 seconds

\- \[ ] api.py /predict logic — load model, preprocess, return prediction + probability

\- \[ ] api.py /health endpoint — input validation, error handling

\- \[ ] Integration test — simulate\_stream.py → api.py end-to-end



\## Week 3: Live Dashboard

\- \[ ] Live Monitoring tab in app.py

\- \[ ] SHAP bar/force plot in live tab

\- \[ ] Auto-refreshing latest reading + prediction display

\- \[ ] Risk gauge / color indicator (healthy → warning → critical)

\- \[ ] Dashboard connected to live /predict API



\## Week 4: Reliability \& Demo Prep

\- \[ ] Prediction logging (CSV/SQLite): timestamp, sensor values, prediction, probability

\- \[ ] Alert system: banner + sound when risk goes high

\- \[ ] History/log viewer tab in dashboard

\- \[ ] API edge-case tests (missing fields, out-of-range values)

\- \[ ] Full demo rehearsal end-to-end

