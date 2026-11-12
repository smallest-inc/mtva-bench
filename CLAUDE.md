# CLAUDE.md

MTVA-Bench: how well an LLM runs a voice agent on a simulated phone call. Code standards are in
`AGENTS.md`.

## Commands

```bash
uv sync                      # install with dev tools
uv run pytest                # tests: no network, no API keys
uv run ruff check            # lint
uv run ruff format           # format (line length 100)
uv run pyright               # type check
uv run mtva-bench --help
uv run mtva-bench data check     # validate and cross-check the bundle
uv run mtva-bench data list [--scenarios] [--agent ID] [--language L] [--difficulty D]
uv run mtva-bench data show <agent_id|scenario_id> [--prompt]
uv run mtva-bench run models/<name>.yaml [--limit N] [--dry-run] [--discard-stale] [--batch]
uv run mtva-bench report runs/<a> [runs/<b> ...] [--by language] [--worst N]
uv run mtva-bench arena runs/<a> runs/<b> [--batch] [--publish]
uv run mtva-bench publish runs/<name>      # its summary into leaderboard/runs/
uv run mtva-bench cost runs/<name> --prices prices.yaml [--publish]   # cost per call, from a price file
uv run mtva-bench leaderboard [--check]    # leaderboard.json, charts and README's marked parts
```

## Runs

A run is a folder, `runs/<name>/` (gitignored). Re-running the same name continues it: done
conversations are skipped, failed ones retried. A conversation made from different inputs
(model spec, caller, scenario, prompt, max turns, seed, `SIMULATOR_VERSION`) stops the run with
the change named; `--discard-stale` redoes them. A `.lock` keeps two runs out of one folder.

## Data

The scenario bundle is a private dataset that lives in `data/<version>/` (gitignored); the
version and its hash come from `benchmark.yaml`. `MTVA_BENCH_DATA_DIR` or `--data` point
elsewhere. Never copy scenario content into this repo: published results hold aggregates only.
`tests/fixtures/bundle/` is a one-agent synthetic bundle;
after editing it, rewrite the `files` hashes in its `bundle.json` (`mtva_bench.data.hash_files`).

## Layout

Each package answers one question and imports only packages listed before it:

| Package | Question |
|---|---|
| `llm/` | how to call an LLM (provider-neutral client, wire protocols, batch backends) |
| `data/` | what a test is (bundle schema, loading, rendering) |
| `simulate/` | how a call happens (agent, mock tools, caller, channel) |
| `capabilities/` | what we measure (one folder per capability, one Verdict per rollout) |
| `scoring/` | how numbers are computed (pure functions) |
| `arena/` | how two runs are compared head to head |
| `run/` | how a run is executed and stored (run folder, stages, manifest) |
| `leaderboard/`, `report.py` | how results are published |

Types live next to the code that produces them. Prompts are `.md` templates next to their code.
