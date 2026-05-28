# Project conventions

This project uses uv for Python environment management.

- Run Python: `uv run python <script>`
- Run pytest: `uv run pytest`
- Add a package: `uv add <package>`
- Add a dev-only package: `uv add --dev <package>`
- Never use `pip install`, `conda install`, or manual venv activation.

Commit pyproject.toml and uv.lock to git.
