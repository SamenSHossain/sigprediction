"""Pull the tournament market list + relationships from the SIG API, map each market to a race_id/party,
crosswalk to Kalshi tickers and Polymarket condition ids by title matching, and write config/races.yaml
(preserving hand edits) and Race/Market rows. Also runs the day-1 probes: fee check, batch-write-count
probe, relationship graph emptiness."""
