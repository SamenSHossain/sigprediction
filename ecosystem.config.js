// pm2 process definitions. Start with: pm2 start ecosystem.config.js
// Every app runs the virtualenv's `sigcup` console script directly (`uv sync` creates .venv/bin/sigcup).
// pm2 resolves `script` as a file path, so it must name that executable (a `script: "-m"` entry fails with
// "Script not found"); `interpreter: "none"` makes pm2 exec it as-is (its shebang is the venv python).
// The trader runs plain `trade`, which follows LIVE from .env: paper unless LIVE=1 AND SIG_API_KEY are set
// (docs/RUNBOOK.md sections 4-5). Never add `--live` here and never pin LIVE in `env`.
const path = require("path");
const ROOT = __dirname;
const SIGCUP = path.join(ROOT, ".venv", "bin", "sigcup");

const common = {
  cwd: ROOT,
  script: SIGCUP,
  interpreter: "none",
  autorestart: true,
  restart_delay: 5000,
  max_restarts: 20,          // a crash loop must not burn the write budget with cancel-all / re-lay cycles
  min_uptime: "30s",
  kill_timeout: 15000,       // give the trader time to cancel resting orders on SIGINT
  env: { PYTHONUNBUFFERED: "1" },
  time: true,
};

module.exports = {
  apps: [
    { ...common, name: "trader",   args: "trade",       out_file: "logs/trader.out.log",   error_file: "logs/trader.err.log" },
    { ...common, name: "feeds",    args: "feeds",       out_file: "logs/feeds.out.log",    error_file: "logs/feeds.err.log" },
    { ...common, name: "reporter", args: "report loop", out_file: "logs/reporter.out.log", error_file: "logs/reporter.err.log" },
  ],
};
