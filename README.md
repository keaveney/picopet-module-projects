# picoPET student guide

An introduction to PET detector light maps, depth-of-interaction reconstruction and uncertainty for final-year undergraduate physics projects at University College Dublin.

- `docs/`, `mkdocs.yml`, `requirements-docs.txt`: editable MkDocs source.
- `project/`: research scripts, teaching inputs and saved results; the current scientific entry points are in `hpc-15-07/`.
- `tools/`: website link checks and teaching-figure helpers.

Read [the student setup instructions](docs/practicals/setup.md) before running the exercises. The supplied research environment uses Python 3.12.4 and `project/requirements-local.txt`; a clean student installation still needs checking on the target machine.

## Preview and edit

Use a separate environment for the website:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

Open http://127.0.0.1:8000. Edit Markdown in `docs/` and navigation in `mkdocs.yml`; the preview reloads when you save. Stop it with Ctrl+C.

## Build and check

With the documentation environment active:

```bash
python -m mkdocs build --strict
python tools/check_site.py
```

The generated website is written to `site/`, which is excluded from Git. Building the website does not require the research environment, PyTorch or GATE. See [maintenance instructions](docs/reference/maintaining.md) for more detail.

## GitHub Pages

This public guide covers PET fundamentals and technical simulation and analysis workflows. Individual research ideas are discussed separately with the students.

The configured website address is https://keaveney.github.io/picopet-module-projects/.

Select **Settings → Pages → Source → GitHub Actions** in the repository. Pushing to `main` then runs `.github/workflows/pages.yml`, which builds the site, checks internal links and publishes only the generated `site/` directory. The first successful deployment is required before the address is available.
