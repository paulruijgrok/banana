# Debugging in Cursor / VS Code

## What went wrong

1. **`-m examples/run_BLI.py`** is invalid.  
   `-m` expects a **Python module** (dotted name), e.g. `banana.cli.run_bli`, not a file path.

2. **`ConnectionRefusedError`** from debugpy often appears when the launch is misconfigured or Cursor runs the **wrong Python** (e.g. `/usr/local/bin/python3` instead of your **conda** env).

## One-time setup (conda env `banana`)

```bash
conda activate banana
cd /path/to/banana
pip install -e .
pip install debugpy   # optional; extension usually bundles it
```

## Point Cursor at conda Python

1. **Command Palette** (`Cmd+Shift+P`) → **Python: Select Interpreter**
2. Choose **`banana`** (e.g. `~/miniconda3/envs/banana/bin/python` or `~/anaconda3/envs/banana/bin/python`).
3. **Reload window** if the debugger still uses `/usr/local/bin/python3`.

Check: bottom-right status bar should show `Python 3.x.x ('banana': conda)`.

## Start the debugger

1. Open **Run and Debug** (sidebar or `Cmd+Shift+D`).
2. Pick a configuration from the dropdown:
   - **Banana: run_BLI.py (example data)** — recommended; runs the example script with paths set.
   - **Banana: module banana.cli.run_bli** — needs `pip install -e .` in `banana` env.
3. Set breakpoints in `src/banana/...` or `examples/run_BLI.py`.
4. Press **F5** or **Start Debugging**.

Do **not** create a launch that uses **Module** with `examples/run_BLI.py`.

## If Connection refused persists

- Fully quit and reopen Cursor.
- In launch.json, use **`"console": "integratedTerminal"`** (already set in this repo).
- Confirm **Run and Debug** uses **Launch** (not Attach).
- Run once in terminal:  
  `conda activate banana && which python`  
  and ensure that path matches the selected interpreter.
