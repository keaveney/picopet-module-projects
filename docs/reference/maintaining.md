# Maintaining this website

The editable content is Markdown under `docs/`, with navigation in `mkdocs.yml`. Figures and diagrams live under `docs/assets/`. The website and research environments are separate: building documentation does not require PyTorch or GATE.

## Read a built site

After building the site, run this from the project root:

```bash
python -m http.server --directory site 8000
```

Open **http://localhost:8000**. Stop the server with Ctrl+C. This serves local files only; it does not publish the website. You can also open `site/index.html` directly, but browser search works best over localhost.

## Rebuild after editing

From the project root, in a separate documentation environment:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python tools/check_site.py
python -m mkdocs serve
```

The documentation requirements pin MkDocs 1.6.1 and Material 9.6.14. Visit **http://127.0.0.1:8000** to see the preview; it reloads when you save changes. The site uses bundled assets and system fonts; its diagrams and equations do not require a third-party CDN.

## Scope of the guide

Keep this public guide focused on PET education and technical simulation and analysis workflows. Individual research questions and unpublished ideas are communicated separately with the students.

## Keeping scientific instructions reliable

Retain the authoritative scripts unchanged unless an explicit scientific change is agreed. Re-run the small procedures after updating source or inputs, save versions and hashes, and update the example data checks. Describe procedures as tested only after executing them in the stated environment.

`tools/make_teaching_figures.py` visualises a supplied light map; it does not estimate position or performance. `tools/check_site.py` checks internal built-site links, anchors and image alt text, and prints its results in the terminal.

## Publishing

MkDocs builds the files in `docs/` into the static website in `site/`. The research scripts and datasets are retained in `project/` alongside the website source.

The website address is **https://keaveney.github.io/picopet-module-projects/**. For the initial deployment, select **Settings → Pages → Build and deployment → Source → GitHub Actions** in the repository.

The workflow in `.github/workflows/pages.yml` builds and checks the website on each push to `main`, then deploys the generated `site/` directory. The **Actions** tab shows progress and any errors; **Publish student guide** can also be run manually from that tab.

For updates, edit locally, preview and check the site, then commit and push the changes:

```bash
python -m mkdocs build --strict
python tools/check_site.py
git status
git add docs mkdocs.yml
git diff --cached
git commit -m "Update student guide"
git push origin main
```

Adjust the paths passed to `git add` for the files you changed. Building or previewing locally does not publish the site; pushing to `main` triggers publication.
