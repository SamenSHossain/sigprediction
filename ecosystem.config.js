// pm2 process definitions. Start with: pm2 start ecosystem.config.js
// All three use the project's virtualenv python (`uv sync` creates .venv).
const path = require("path");
const ROOT = __dirname;
const PY = path.join(ROOT, ".venv", "bin", "python");

const common = {
  cwd: ROOT,
  interpreter: PY,
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
    { ...common, name: "trader",   script: "-m", args: "sigcup.cli trade",  out_file: "logs/trader.out.log",   error_file: "logs/trader.err.log" },
    { ...common, name: "feeds",    script: "-m", args: "sigcup.cli feeds",  out_file: "logs/feeds.out.log",    error_file: "logs/feeds.err.log" },
    { ...common, name: "reporter", script: "-m", args: "sigcup.cli report loop", out_file: "logs/reporter.out.log", error_file: "logs/reporter.err.log" },
  ],
};
