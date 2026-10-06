# Contributing

1. Fork and create a branch from `main`.
2. `pip install -r requirements.txt` (and `server/requirements.txt` for the API).
3. Keep changes focused. Match the style of the surrounding code.
4. Before opening a PR: `python -m compileall -q server experiments *.py` and, for server changes,
   `python -c "import server.app"`.
5. Do not commit secrets, `.env` files, or IBM tokens. Do not commit large regenerated datasets
   unless the change is about the data; say in the PR how they were generated (script, seeds, circuit counts).

Open an issue first for large changes (new models, changes to published benchmark definitions).
