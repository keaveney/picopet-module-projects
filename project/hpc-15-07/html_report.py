from html import escape
from pathlib import Path

import numpy as np


def _humanize_plot_name(path: Path):
    name = path.stem.replace("_", "-").replace("-", " ")
    return " ".join(part.capitalize() for part in name.split())


def _format_metric_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return escape(str(value))


def _collect_scalar_metrics(metrics, prefix=""):
    rows = []
    if not isinstance(metrics, dict):
        return rows

    for key, value in metrics.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_collect_scalar_metrics(value, label))
        elif isinstance(value, (float, int, np.floating, np.integer, str)):
            rows.append((label, value))
    return rows


def write_html_report(outdir, metrics=None, filename="index.html"):
    """Write a static clickable gallery for evaluation plots in outdir."""
    outdir = Path(outdir)
    png_paths = sorted(outdir.rglob("*.png"), key=lambda p: p.relative_to(outdir).as_posix())
    metric_rows = _collect_scalar_metrics(metrics) if metrics is not None else []

    sections = {}
    for path in png_paths:
        rel_parent = path.parent.relative_to(outdir)
        section = "Main plots" if rel_parent == Path(".") else rel_parent.as_posix()
        sections.setdefault(section, []).append(path)

    def image_card(path):
        rel = path.relative_to(outdir).as_posix()
        title = _humanize_plot_name(path)
        return f"""
        <article class="card">
          <a class="plot-link" href="{escape(rel)}" data-full="{escape(rel)}" data-title="{escape(title)}">
            <img src="{escape(rel)}" alt="{escape(title)}" loading="lazy">
          </a>
          <div class="caption">{escape(title)}</div>
          <div class="path">{escape(rel)}</div>
        </article>
        """

    section_html = []
    for idx, (section, paths) in enumerate(sections.items()):
        open_attr = " open" if idx == 0 else ""
        cards = "\n".join(image_card(path) for path in paths)
        section_html.append(
            f"""
            <details class="section"{open_attr}>
              <summary>{escape(section)} <span>{len(paths)} plots</span></summary>
              <div class="grid">{cards}</div>
            </details>
            """
        )

    metrics_html = ""
    if metric_rows:
        preview_rows = metric_rows[:80]
        metrics_html = """
        <details class="section">
          <summary>Scalar metrics <span>{n} shown</span></summary>
          <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
              {rows}
            </tbody>
          </table>
        </details>
        """.format(
            n=len(preview_rows),
            rows="\n".join(
                f"<tr><td>{escape(label)}</td><td>{_format_metric_value(value)}</td></tr>"
                for label, value in preview_rows
            ),
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CNN Evaluation Report</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --ink: #1f2528;
      --muted: #667075;
      --line: #d8d0c3;
      --card: #fffdf8;
      --accent: #1f5d52;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, #fffdf8, #ece4d8);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: -0.03em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      padding: 22px 28px 40px;
    }}
    .section {{
      margin: 0 0 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 253, 248, 0.72);
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      padding: 14px 18px;
      font-weight: 700;
      color: var(--accent);
    }}
    summary span {{
      color: var(--muted);
      font-weight: 500;
      margin-left: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 18px;
      padding: 0 18px 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      box-shadow: 0 10px 26px rgba(31, 37, 40, 0.06);
    }}
    .card img {{
      width: 100%;
      height: 220px;
      object-fit: contain;
      background: white;
      border-radius: 8px;
      border: 1px solid #eee7dc;
    }}
    .plot-link {{
      display: block;
      cursor: zoom-in;
    }}
    .caption {{
      margin-top: 9px;
      font-weight: 650;
      font-size: 14px;
    }}
    .path {{
      margin-top: 4px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      word-break: break-all;
    }}
    table {{
      width: calc(100% - 36px);
      margin: 0 18px 18px;
      border-collapse: collapse;
      font-size: 13px;
      background: var(--card);
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 7px 9px;
      text-align: left;
    }}
    th {{
      background: #eee7dc;
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(17, 23, 25, 0.86);
      z-index: 1000;
      padding: 28px;
    }}
    .lightbox.open {{
      display: flex;
    }}
    .lightbox-panel {{
      position: relative;
      width: min(96vw, 1500px);
      height: min(92vh, 1000px);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .lightbox-title {{
      color: #fffdf8;
      font-weight: 700;
      font-size: 16px;
      line-height: 1.3;
      padding-right: 48px;
    }}
    .lightbox img {{
      flex: 1;
      min-height: 0;
      object-fit: contain;
      background: white;
      border-radius: 10px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
    }}
    .lightbox-close {{
      position: absolute;
      top: -10px;
      right: 0;
      width: 38px;
      height: 38px;
      border: 1px solid rgba(255, 253, 248, 0.45);
      border-radius: 999px;
      background: rgba(255, 253, 248, 0.12);
      color: #fffdf8;
      font-size: 26px;
      line-height: 32px;
      cursor: pointer;
    }}
    .lightbox-help {{
      color: rgba(255, 253, 248, 0.78);
      font-size: 12px;
      text-align: right;
    }}
  </style>
</head>
<body>
  <header>
    <h1>CNN Evaluation Report</h1>
    <div class="meta">{len(png_paths)} plots found in <code>{escape(str(outdir))}</code></div>
  </header>
  <main>
    {metrics_html}
    {''.join(section_html) if section_html else '<p>No PNG plots found.</p>'}
  </main>
  <div class="lightbox" id="plot-lightbox" aria-hidden="true">
    <div class="lightbox-panel">
      <button class="lightbox-close" type="button" aria-label="Close preview">&times;</button>
      <div class="lightbox-title" id="lightbox-title"></div>
      <img id="lightbox-image" src="" alt="">
      <div class="lightbox-help">Click outside the image or press Escape to close</div>
    </div>
  </div>
  <script>
    const lightbox = document.getElementById("plot-lightbox");
    const lightboxImage = document.getElementById("lightbox-image");
    const lightboxTitle = document.getElementById("lightbox-title");
    const closeButton = lightbox.querySelector(".lightbox-close");

    function openLightbox(src, title) {{
      lightboxImage.src = src;
      lightboxImage.alt = title;
      lightboxTitle.textContent = title;
      lightbox.classList.add("open");
      lightbox.setAttribute("aria-hidden", "false");
      document.body.style.overflow = "hidden";
    }}

    function closeLightbox() {{
      lightbox.classList.remove("open");
      lightbox.setAttribute("aria-hidden", "true");
      lightboxImage.src = "";
      document.body.style.overflow = "";
    }}

    document.querySelectorAll(".plot-link").forEach((link) => {{
      link.addEventListener("click", (event) => {{
        event.preventDefault();
        openLightbox(link.dataset.full, link.dataset.title);
      }});
    }});

    closeButton.addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", (event) => {{
      if (event.target === lightbox) {{
        closeLightbox();
      }}
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape" && lightbox.classList.contains("open")) {{
        closeLightbox();
      }}
    }});
  </script>
</body>
</html>
"""

    report_path = outdir / filename
    report_path.write_text(html)
    return report_path
