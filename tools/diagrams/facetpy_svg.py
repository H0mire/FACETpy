"""
FACETpy diagram builder — the shared visual toolkit for on-brand SVG diagrams.

Everything here is derived from the FACETpy logo (brain gradient + EEG plot on
white). Import these primitives instead of hand-rolling shapes so every diagram
— class, flow, architecture, sequence — reads as one coherent system.

Stdlib only (math, html, subprocess). Run with `uv run python`.

Typical use
-----------
    from facetpy_svg import Diagram, class_box, edge, side

    d = Diagram(1030, 660, title="Restaurant", subtitle="domain model")
    cust = class_box(70, 120, 220, "Customer",
                     [("-", "name", "str")], [("+", "placeOrder()", "Order")])
    order = class_box(70, 400, 230, "Order", [("-", "id", "int")], [])
    d.add(cust, order)
    d.add_edge(edge([side(cust, "bottom"), side(order, "top")],
                    label=("places", 190, 340), mult=[("1", 190, 320)]))
    d.save("out.svg")          # writes SVG
    d.render_png("out.png")    # rasterises a preview via rsvg-convert
"""
from __future__ import annotations
import html
import math
import shutil
import subprocess

# --------------------------------------------------------------------------- #
#  Design tokens — the single source of truth for the FACETpy look.
# --------------------------------------------------------------------------- #
FONT = "'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', sans-serif"

LIGHT = {
    "ink":      "#173147",  # wordmark navy — text + strong strokes
    "navy_d":   "#0f2537",  # darkest navy (header gradient top)
    "blue":     "#2f6d94",  # mid blue (header gradient bottom, accents)
    "blue400":  "#7eb8d2",  # brain left hemisphere
    "blue200":  "#aedbec",  # brain right hemisphere
    "slate":    "#5b7488",  # muted type / secondary text
    "surface":  "#ffffff",  # card fill
    "tint":     "#eef6fb",  # faint surface tint / chip fill
    "bg_a":     "#ffffff",  # canvas gradient start
    "bg_b":     "#eef6fb",  # canvas gradient end
    "grid":     "#173147",  # grid line colour
    "grid_op":  "0.045",    # grid line opacity
    "border_op":"0.16",     # card border opacity
    "header_fg":"#ffffff",  # text on header bar
    "pos":      "#3f9d5a",  # visibility "+" (public)
    "neg":      "#c05a5a",  # visibility "-" (private)
    "prot":     "#2f6d94",  # visibility "#" (protected)
}

DARK = {
    "ink":      "#e8f2f8",
    "navy_d":   "#0b1c2b",
    "blue":     "#4f97bd",
    "blue400":  "#7eb8d2",
    "blue200":  "#aedbec",
    "slate":    "#9db6c8",
    "surface":  "#16324a",
    "tint":     "#1c3c56",
    "bg_a":     "#0d2233",
    "bg_b":     "#12293c",
    "grid":     "#aedbec",
    "grid_op":  "0.05",
    "border_op":"0.35",
    "header_fg":"#ffffff",
    "pos":      "#67c98a",
    "neg":      "#e08a8a",
    "prot":     "#7eb8d2",
}

C = dict(LIGHT)  # active palette; Diagram(theme=...) swaps this.


def set_theme(name: str) -> None:
    C.clear()
    C.update(DARK if name == "dark" else LIGHT)


# --------------------------------------------------------------------------- #
#  Layout constants (keep members aligned across every card).
# --------------------------------------------------------------------------- #
ROW = 23    # line height inside a compartment
HEAD = 36   # header-bar height
PAD = 11    # compartment vertical padding

# Fixed canvas width for every diagram. All diagrams share this width so they
# scale identically when embedded, keeping the title block the same visual size
# across documents. Lay content out within [0, CANVAS_W]; only the height varies.
CANVAS_W = 1000
TITLE_MARGIN = 90   # top space reserved when a title block is present


def esc(s) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- #
#  Low-level shape helpers
# --------------------------------------------------------------------------- #
def _rounded_top(x, y, w, head, r):
    """Path for a rectangle whose TOP corners only are rounded (header bar)."""
    return (f'M{x} {y+r} A{r} {r} 0 0 1 {x+r} {y} '
            f'L{x+w-r} {y} A{r} {r} 0 0 1 {x+w} {y+r} '
            f'L{x+w} {y+head} L{x} {y+head} Z')


def _node_dot(cx, cy):
    """The connectome dot from the logo's brain — the header signature mark."""
    return (f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="{C["blue200"]}" '
            f'stroke="#ffffff" stroke-opacity="0.85" stroke-width="1.2"/>')


def _text(x, y, s, size=12.5, fill=None, weight=None, anchor="start",
          italic=False, opacity=None):
    fill = fill or C["ink"]
    extra = ""
    if weight:  extra += f' font-weight="{weight}"'
    if italic:  extra += ' font-style="italic"'
    if anchor != "start": extra += f' text-anchor="{anchor}"'
    if opacity is not None: extra += f' fill-opacity="{opacity}"'
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}"{extra}>{esc(s)}</text>')


def _text_w(s, size=11):
    """Estimate rendered text width. Rough, but accounts for wide glyphs
    (guillemets «», m/w, caps) and narrow ones (i/l/punctuation) so auto-sized
    chips actually contain their text instead of clipping it."""
    w = 0.0
    for ch in s:
        if ch in "«»mwMW—@":            w += size * 0.92
        elif ch in "iljtfIrÎ.,:;'|!":   w += size * 0.34
        elif ch.isupper() or ch.isdigit(): w += size * 0.66
        else:                            w += size * 0.55
    return w


def _header(x, y, w, name, stereotype=None):
    """Gradient header bar + connectome dot + title."""
    p = [f'<path d="{_rounded_top(x, y, w, HEAD, 12)}" fill="url(#fp-header)"/>',
         _node_dot(x + 18, y + HEAD / 2)]
    if stereotype:
        p.append(_text(x + 34, y + HEAD / 2 - 4, f"«{stereotype}»", size=10.5,
                       fill=C["header_fg"], italic=True, opacity=0.7))
        p.append(_text(x + 34, y + HEAD / 2 + 11, name, size=15,
                       fill=C["header_fg"], weight=600))
    else:
        p.append(_text(x + 34, y + HEAD / 2 + 5, name, size=15.5,
                       fill=C["header_fg"], weight=600))
    return "".join(p)


def _card_shell(x, y, w, h):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{C["surface"]}" stroke="{C["ink"]}" '
            f'stroke-opacity="{C["border_op"]}" stroke-width="1.25" '
            f'filter="url(#fp-shadow)"/>')


def _divider(x, y, w):
    return (f'<line x1="{x}" y1="{y}" x2="{x+w}" y2="{y}" '
            f'stroke="{C["ink"]}" stroke-opacity="0.12"/>')


def _geom(svg, x, y, w, h):
    """Bundle an svg fragment with geometry + edge anchors."""
    return {
        "svg": svg, "x": x, "y": y, "w": w, "h": h,
        "cx": x + w / 2, "cy": y + h / 2,
        "top": (x + w / 2, y), "bottom": (x + w / 2, y + h),
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
    }


def side(box, name):
    """Anchor point on a box: 'top' | 'bottom' | 'left' | 'right' | 'center'."""
    if name == "center":
        return (box["cx"], box["cy"])
    return box[name]


# --------------------------------------------------------------------------- #
#  Nodes
# --------------------------------------------------------------------------- #
def _member(x, y, w, vis, name, typ, kind="attr"):
    vis_color = {"+": C["pos"], "-": C["neg"], "#": C["prot"]}.get(vis, C["prot"])
    s = (f'<text x="{x+14}" y="{y}" font-family="{FONT}" font-size="12.5" '
         f'fill="{C["ink"]}">'
         f'<tspan fill="{vis_color}" font-weight="600">{esc(vis)} </tspan>'
         f'<tspan fill="{C["ink"]}">{esc(name)}</tspan>')
    if typ:
        s += f'<tspan fill="{C["slate"]}">: {esc(typ)}</tspan>'
    return s + "</text>"


def class_box(x, y, w, name, attrs, methods, stereotype=None):
    """UML class card. attrs/methods are lists of (visibility, name, type).

    visibility is one of '+', '-', '#'. Height is computed from the row counts
    so members always line up. Returns a geometry dict (pass to add / side)."""
    attr_h = PAD * 2 + max(len(attrs), 1) * ROW
    meth_h = PAD * 2 + max(len(methods), 1) * ROW
    h = HEAD + attr_h + meth_h
    p = [_card_shell(x, y, w, h), _header(x, y, w, name, stereotype)]
    cy = y + HEAD
    p.append(_divider(x, cy, w))
    ay = cy + PAD + 15
    for vis, mname, typ in attrs:
        p.append(_member(x, ay, w, vis, mname, typ)); ay += ROW
    dy = cy + attr_h
    p.append(_divider(x, dy, w))
    my = dy + PAD + 15
    for vis, mname, typ in methods:
        p.append(_member(x, my, w, vis, mname, typ, kind="method")); my += ROW
    return _geom("".join(p), x, y, w, h)


def card(x, y, w, h, title, lines=None, stereotype=None):
    """Generic titled card for architecture / package / component boxes.
    `lines` is a list of plain strings shown in the body.

    `h` is a MINIMUM: if the body lines need more room than `h` allows, the card
    grows to fit so text never spills past the border (a header-only card keeps
    exactly `h`). Read the real height back from the returned geometry dict."""
    if lines:
        needed = HEAD + PAD * 2 + len(lines) * ROW
        h = max(h, needed)
    p = [_card_shell(x, y, w, h), _header(x, y, w, title, stereotype)]
    if lines:
        p.append(_divider(x, y + HEAD, w))
        ly = y + HEAD + PAD + 15
        for ln in lines:
            p.append(_text(x + 16, ly, ln, size=12.5, fill=C["slate"])); ly += ROW
    return _geom("".join(p), x, y, w, h)


def pill(x, y, w, h, text, accent=None):
    """Flow process node: soft rounded rect, tinted, centred label."""
    accent = accent or C["blue"]
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2 if h<40 else 12}" '
           f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
           f'stroke-width="1.5" filter="url(#fp-shadow)"/>'
           f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>'
           + _text(x + w / 2, y + h / 2 + 4.5, text, size=13, weight=500,
                   anchor="middle"))
    return _geom(svg, x, y, w, h)


def capsule(x, y, w, h, text):
    """Flow terminator (start/end): fully rounded stadium in the header gradient."""
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" '
           f'fill="url(#fp-header)" filter="url(#fp-shadow)"/>'
           + _text(x + w / 2, y + h / 2 + 4.5, text, size=13, weight=600,
                   fill=C["header_fg"], anchor="middle"))
    return _geom(svg, x, y, w, h)


def decision(cx, cy, w, h, text):
    """Flow decision diamond."""
    pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    svg = (f'<polygon points="{pts}" fill="{C["surface"]}" stroke="{C["blue"]}" '
           f'stroke-width="1.5" filter="url(#fp-shadow)"/>'
           + _text(cx, cy + 4.5, text, size=12, weight=500, anchor="middle"))
    return _geom(svg, cx - w / 2, cy - h / 2, w, h)


def container(x, y, w, h, title):
    """Grouping frame (swimlane / package / subsystem) with a title tab."""
    tab_w = 20 + len(title) * 7.5
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
           f'fill="{C["ink"]}" fill-opacity="0.03" stroke="{C["blue"]}" '
           f'stroke-opacity="0.4" stroke-width="1.4" stroke-dasharray="2 5"/>'
           f'<path d="{_rounded_top(x, y, tab_w, 26, 14)}" fill="url(#fp-header)"/>'
           + _text(x + 14, y + 17.5, title, size=12.5, weight=600,
                   fill=C["header_fg"]))
    return _geom(svg, x, y, w, h)


def note(x, y, w, h, text):
    """Annotation / sticky note with a folded corner."""
    fold = 14
    svg = (f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} L{x+w} {y+h} '
           f'L{x} {y+h} Z" fill="{C["tint"]}" stroke="{C["slate"]}" '
           f'stroke-opacity="0.4" stroke-width="1"/>'
           f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} L{x+w} {y+fold}" '
           f'fill="none" stroke="{C["slate"]}" stroke-opacity="0.4"/>'
           + _text(x + 10, y + 20, text, size=11.5, fill=C["slate"]))
    return _geom(svg, x, y, w, h)


# --------------------------------------------------------------------------- #
#  Sequence-diagram helpers
# --------------------------------------------------------------------------- #
def lifeline(cx, top, bottom, name, w=140, head=HEAD):
    """Actor/object header card + dashed lifeline down to `bottom`."""
    x = cx - w / 2
    svg = (f'<path d="M{cx} {top+head} L{cx} {bottom}" stroke="{C["ink"]}" '
           f'stroke-opacity="0.35" stroke-width="1.3" stroke-dasharray="3 5"/>'
           + _card_shell(x, top, w, head)
           + f'<path d="{_rounded_top(x, top, w, head, 12)}" fill="url(#fp-header)"/>'
           + _node_dot(x + 16, top + head / 2)
           + _text(cx + 6, top + head / 2 + 5, name, size=13.5, weight=600,
                   fill=C["header_fg"], anchor="middle"))
    return _geom(svg, x, top, w, bottom - top)


def activation(cx, y1, y2, w=10):
    return (f'<rect x="{cx-w/2}" y="{y1}" width="{w}" height="{y2-y1}" rx="2" '
            f'fill="{C["blue200"]}" stroke="{C["blue"]}" stroke-width="1"/>')


def message(x1, x2, y, text, dashed=False, ret=False):
    """Horizontal message arrow between two lifelines with a label above it."""
    dash = ' stroke-dasharray="4 4"' if (dashed or ret) else ""
    marker = "fp-arrow-open" if ret else "fp-arrow"
    line = (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{C["ink"]}" '
            f'stroke-opacity="0.6" stroke-width="1.5" marker-end="url(#{marker})"{dash}/>')
    lx = (x1 + x2) / 2
    return line + _text(lx, y - 7, text, size=11.5, fill=C["blue"], weight=500,
                        anchor="middle")


# --------------------------------------------------------------------------- #
#  Connectors
# --------------------------------------------------------------------------- #
def edge(points, marker_end="arrow", marker_start=None, label=None, mult=None):
    """Orthogonal/straight connector through `points` [(x,y), ...].

    marker_* ∈ {'arrow','arrow-open','diamond','diamond-open','triangle', None}.
    label = (text, x, y) draws a chip CENTERED on (x, y); the chip auto-sizes to
    the text so it never clips (guillemets/caps included). mult = [(text, x, y),
    ...] multiplicities (drawn left-anchored at each point)."""
    pstr = " ".join(f"{px},{py}" for px, py in points)
    a = f'fill="none" stroke="{C["ink"]}" stroke-opacity="0.55" stroke-width="1.5"'
    if marker_end:   a += f' marker-end="url(#fp-{marker_end})"'
    if marker_start: a += f' marker-start="url(#fp-{marker_start})"'
    out = [f'<polyline points="{pstr}" {a}/>']
    if label:
        txt, lx, ly = label
        bw = _text_w(txt, 11) + 14           # chip auto-sizes to the text
        out.append(f'<rect x="{lx-bw/2:.1f}" y="{ly-12}" width="{bw:.1f}" '
                   f'height="17" rx="4" fill="{C["tint"]}" stroke="{C["ink"]}" '
                   f'stroke-opacity="0.10"/>')
        out.append(_text(lx, ly, txt, size=11, fill=C["blue"], weight=500,
                         anchor="middle"))
    for mtxt, mx, my in (mult or []):
        out.append(_text(mx, my, mtxt, size=10.5, fill=C["slate"]))
    return "".join(out)


# --------------------------------------------------------------------------- #
#  Brand furniture: title block, EEG wave, footer, background, defs
# --------------------------------------------------------------------------- #
def eeg_wave(x, y, w, color=None, opacity=0.9):
    """The FACETpy signature: flat → ripple → spike → flat EEG trace."""
    color = color or C["blue400"]
    n, pts = 60, []
    for i in range(n + 1):
        t = i / n
        px = x + t * w
        if t < 0.25:   py = y
        elif t < 0.55: py = y - 6 * math.sin((t - 0.25) / 0.30 * math.pi * 3)
        elif t < 0.68: py = y - 26 * math.sin((t - 0.55) / 0.13 * math.pi)
        else:          py = y
        pts.append(f"{px:.1f},{py:.1f}")
    return (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-opacity="{opacity}" stroke-width="2" stroke-linejoin="round"/>')


def title_block(title, subtitle=None, x=70, y=52):
    p = [f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="24" '
         f'font-weight="700" fill="{C["ink"]}">{esc(title)}']
    if subtitle:
        p.append(f' <tspan font-weight="400" fill="{C["slate"]}" '
                 f'font-size="15">· {esc(subtitle)}</tspan>')
    p.append("</text>")
    uw = 22 + len(title) * 9
    p.append(f'<rect x="{x}" y="{y+10}" width="{uw}" height="3" rx="1.5" '
             f'fill="url(#fp-header)"/>')
    # Place the signature wave clear of the full title text (title + subtitle).
    text_end = x + len(title) * 13.5 + (len(subtitle) + 3) * 8 * bool(subtitle)
    p.append(eeg_wave(text_end + 30, y - 8, 90))
    return "".join(p)


def footer(W, H, text="Made with FACETpy"):
    return _text(W - 20, H - 18, text, size=11, fill=C["slate"],
                 anchor="end", opacity=0.8)


def defs():
    return f'''<defs>
  <linearGradient id="fp-header" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="{C["navy_d"]}"/><stop offset="1" stop-color="{C["blue"]}"/>
  </linearGradient>
  <linearGradient id="fp-bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{C["bg_a"]}"/><stop offset="1" stop-color="{C["bg_b"]}"/>
  </linearGradient>
  <pattern id="fp-grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M40 0 L0 0 0 40" fill="none" stroke="{C["grid"]}" stroke-opacity="{C["grid_op"]}" stroke-width="1"/>
  </pattern>
  <filter id="fp-shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="{C["navy_d"]}" flood-opacity="0.13"/>
  </filter>
  <marker id="fp-arrow" markerWidth="12" markerHeight="12" refX="9" refY="5" orient="auto">
    <path d="M1 1 L9 5 L1 9" fill="none" stroke="{C["ink"]}" stroke-opacity="0.7" stroke-width="1.5"/>
  </marker>
  <marker id="fp-arrow-open" markerWidth="13" markerHeight="12" refX="10" refY="5" orient="auto">
    <path d="M1 1 L10 5 L1 9" fill="none" stroke="{C["ink"]}" stroke-opacity="0.6" stroke-width="1.3"/>
  </marker>
  <marker id="fp-diamond" markerWidth="18" markerHeight="12" refX="1" refY="5" orient="auto">
    <path d="M1 5 L8 1.5 L15 5 L8 8.5 Z" fill="{C["ink"]}" fill-opacity="0.75" stroke="{C["ink"]}" stroke-width="1"/>
  </marker>
  <marker id="fp-diamond-open" markerWidth="18" markerHeight="12" refX="1" refY="5" orient="auto">
    <path d="M1 5 L8 1.5 L15 5 L8 8.5 Z" fill="{C["surface"]}" stroke="{C["ink"]}" stroke-width="1.3"/>
  </marker>
  <marker id="fp-triangle" markerWidth="16" markerHeight="14" refX="13" refY="6" orient="auto">
    <path d="M1 1 L13 6 L1 11 Z" fill="{C["surface"]}" stroke="{C["ink"]}" stroke-width="1.4"/>
  </marker>
</defs>'''


# --------------------------------------------------------------------------- #
#  Public low-level building blocks — for creative / custom elements.
#
#  When you need something the node primitives don't offer (a special arrow, an
#  icon, a representative glyph, a bespoke shape), draw raw SVG but build it from
#  THESE so it stays on-brand: pull colours from `C[...]`, use `FONT`, and reuse
#  the shared defs via url(#fp-shadow) / url(#fp-header) / url(#fp-grid) and the
#  markers. Register your own markers/gradients/symbols with Diagram.add_defs().
# --------------------------------------------------------------------------- #
text = _text                # text(x, y, s, size=, fill=, weight=, anchor=, ...)
text_width = _text_w        # estimate rendered width for auto-sizing chips/boxes
node_dot = _node_dot        # the logo connectome dot: node_dot(cx, cy)
rounded_top = _rounded_top  # path string for a top-rounded rect (header shape)
card_shell = _card_shell    # the branded rounded card + border + shadow
divider = _divider          # hairline compartment separator


# --------------------------------------------------------------------------- #
#  Diagram assembler
# --------------------------------------------------------------------------- #
class Diagram:
    """Accumulates edges (drawn first) and nodes (drawn on top), then wraps
    them with the shared defs, background, title block and footer."""

    def __init__(self, height, width=CANVAS_W, title=None, subtitle=None,
                 grid=True, theme="light", footer_text="Made with FACETpy",
                 background=False):
        set_theme(theme)
        # Width is fixed by default (CANVAS_W) so every diagram scales the same
        # when placed in a document; only pass height. Override width only if you
        # deliberately need a different fixed size for a whole diagram family.
        self.width, self.height = width, height
        self.title, self.subtitle = title, subtitle
        self.grid, self.footer_text = grid, footer_text
        self.background = background   # False => transparent canvas
        self._edges, self._nodes, self._extra_defs = [], [], []

    @staticmethod
    def _frag(item):
        return item["svg"] if isinstance(item, dict) else item

    def add(self, *items):
        """Add node-layer fragments (cards, pills, ...). Drawn above edges."""
        self._nodes.extend(self._frag(i) for i in items)
        return self

    def add_edge(self, *items):
        """Add connector-layer fragments. Drawn below nodes."""
        self._edges.extend(self._frag(i) for i in items)
        return self

    def add_defs(self, *snippets):
        """Register extra <defs> content — custom <marker>, <linearGradient>,
        <filter>, <symbol>, etc. — so bespoke raw-SVG elements can reference them
        via url(#your-id) exactly like the built-in fp-* defs. Give your ids an
        `fp-` prefix to signal they belong to the FACETpy system."""
        self._extra_defs.extend(snippets)
        return self

    def render_svg(self) -> str:
        W, H = self.width, self.height
        d = defs()
        if self._extra_defs:
            d = d.replace("</defs>", "\n".join(self._extra_defs) + "\n</defs>")
        body = [d]
        # Transparent by default; only paint a canvas when explicitly requested.
        if self.background:
            body.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#fp-bg)"/>')
        if self.grid:
            body.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#fp-grid)"/>')
        if self.title:
            body.append(title_block(self.title, self.subtitle))
        body.extend(self._edges)
        body.extend(self._nodes)
        if self.footer_text:
            body.append(footer(W, H, self.footer_text))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
                f'width="{W}" height="{H}" font-family="{FONT}">\n'
                + "\n".join(body) + "\n</svg>\n")

    def save(self, path) -> str:
        with open(path, "w") as f:
            f.write(self.render_svg())
        return path

    def render_png(self, png_path, svg_path=None, width=1280) -> str:
        """Rasterise a preview. Writes the SVG first if svg_path is given,
        otherwise assumes it is already saved next to png_path."""
        if svg_path:
            self.save(svg_path)
        else:
            svg_path = png_path.rsplit(".", 1)[0] + ".svg"
            self.save(svg_path)
        exe = shutil.which("rsvg-convert")
        if not exe:
            raise RuntimeError("rsvg-convert not found — install librsvg "
                               "(`brew install librsvg`) or render the SVG another way.")
        subprocess.run([exe, "-w", str(width), svg_path, "-o", png_path], check=True)
        return png_path
