"""The standalone page a single widget is previewed on.

Each widget in ``apps/web`` has a demo module that writes one self-contained HTML
file so the widget can be opened on its own, away from the site. Those modules
carried five copies of the same shell: a ``_PAGE`` template, a ``build_html``, a
``save_demo`` and a ``main``, about 150 lines, with the CSS character-identical
in all five. What actually differed per module was a title, a heading, a
standfirst, one or two container divs, and sometimes a closing note.

So that is what this takes. The demo modules keep the part ``build_site`` imports
from them -- their ``*_bundle`` and ``*_mount`` pair -- and describe the page in
six strings.

Deliberately *not* shared with :func:`apps.build_site._document`. The site's
shell carries a topbar, a theme toggle, an in-page index, a hand-off card and a
footer, and its stylesheet is 260 lines; a preview page is a heading and a widget.
They are different chrome, and merging them would mean one of the two carrying
the other's baggage.
"""
from __future__ import annotations

import os

#: Shared across every preview page. Small on purpose: a preview is scaffolding
#: for looking at one widget, and its chrome should not compete with the widget.
#: Dark-mode rules follow the OS preference only -- there is no theme toggle here
#: to honour, unlike on the site.
CSS = """  :root{color-scheme:light dark;}
  body{margin:0;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:#fff;color:#1b1f24;}
  @media (prefers-color-scheme:dark){body{background:#0d1117;color:#e6eaf0;}}
  .wrap{max-width:880px;margin:0 auto;padding:32px 22px 56px;}
  h1{font-size:1.5rem;margin:0 0 6px;}
  .sub{color:#5a6472;margin:0 0 22px;}
  @media (prefers-color-scheme:dark){.sub{color:#9aa6b5;}}
  .note{margin-top:26px;font-size:13px;color:#5a6472;border-top:1px solid #d7dde5;padding-top:14px;}
  @media (prefers-color-scheme:dark){.note{color:#9aa6b5;border-color:#30363d;}}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;}"""


def preview_page(*, title: str, heading: str, standfirst: str, hosts,
                 bundle: str, mount: str, note: str = None) -> str:
    """Return one standalone preview page as HTML.

    Parameters
    ----------
    title : browser title, without the ``photonn --`` prefix this adds.
    heading : the ``<h1>``.
    standfirst : the paragraph under it. May contain markup.
    hosts : container ids the widget mounts into, in document order.
    bundle, mount : the ``<script>`` blocks from the widget's own demo module.
    note : optional closing paragraph, for a caveat worth stating on the page.
    """
    if isinstance(hosts, str):
        hosts = (hosts,)
    divs = "\n  ".join(f'<div id="{host}"></div>' for host in hosts)
    tail = f'\n  <p class="note">{note}</p>' if note else ""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>photonn &mdash; {title}</title>\n"
        f"<style>\n{CSS}\n</style>\n"
        "</head>\n<body>\n"
        '<div class="wrap">\n'
        f"  <h1>{heading}</h1>\n"
        f'  <p class="sub">{standfirst}</p>\n'
        f"  {divs}{tail}\n"
        "</div>\n"
        f"{bundle}\n{mount}\n"
        "</body>\n</html>\n"
    )


def save_preview(path: str, html: str) -> str:
    """Write a preview page and return its path.

    ``newline="\\n"`` for the same reason ``build_site.main`` uses it: these files
    are committed, ``.gitattributes`` normalises them to LF, and without it the
    working tree churns on every regeneration.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    return path


def default_path(module_file: str, name: str) -> str:
    """``apps/<name>.html``, beside the demo module that asked for it."""
    return os.path.join(os.path.dirname(os.path.abspath(module_file)), f"{name}.html")
