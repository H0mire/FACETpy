# FACETpy Diagram Design System

This is the full visual specification for FACETpy diagrams. You (the subagent)
are building an SVG that must look like it belongs to FACETpy — the same brand
world as the logo (a blue-gradient brain wired with connectome nodes, next to an
EEG trace on a plotting grid). Nothing here is pulled from older diagrams; the
whole language derives from the logo.

**Start from the toolkit; extend it when the diagram calls for it.**
`assets/facetpy_svg.py` encodes every token and shape below — compose from its
primitives so output stays consistent across runs. But the toolkit is a
foundation, not a cage: when the diagram genuinely needs something it doesn't
offer (a special arrow, an icon, a representative glyph, a bespoke shape), you
are encouraged to draw it yourself. The one rule is **stay in the brand system**
— build custom elements from the tokens and shared defs, not ad-hoc colours.
Section 7 shows exactly how.

---

## 1. Brand palette (derived from the logo)

| Token | Light | Meaning |
|-------|-------|---------|
| `ink` | `#173147` | wordmark navy — body text, strong strokes |
| `navy_d` | `#0f2537` | darkest navy — top of the header gradient |
| `blue` | `#2f6d94` | mid blue — bottom of gradient, accents, labels |
| `blue400` | `#7eb8d2` | brain left hemisphere — the EEG-wave colour |
| `blue200` | `#aedbec` | brain right hemisphere — the connectome dot |
| `slate` | `#5b7488` | muted secondary text, type annotations |
| `surface` | `#ffffff` | card fill |
| `tint` | `#eef6fb` | faint surface / chip fill |

Semantic member colours: `pos` `#3f9d5a` (`+` public), `neg` `#c05a5a`
(`-` private), `prot` `#2f6d94` (`#` protected). A dark theme (`theme="dark"`)
is defined in the toolkit; use it only when asked.

Typography is one family everywhere:
`'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', sans-serif`.

---

## 2. What makes a diagram read as "FACETpy"

Five signature moves — keep all of them:

1. **Soft cards.** Rounded corners (r≈12), white fill, a navy hairline border at
   ~16% opacity, and a subtle drop shadow. Never hard-edged boxes.
2. **Gradient header bar.** Every titled node has a navy→blue gradient header
   (`url(#fp-header)`) with the title in white, semibold.
3. **Connectome dot.** A small `blue200` dot with a white ring sits at the left
   of each header — the node motif from the logo's brain.
4. **Plotting grid + EEG wave.** A faint 40px grid fills the canvas (the logo's
   EEG plot paper). A single EEG trace (flat → ripple → spike → flat) accents the
   title block — the brand's heartbeat.
5. **Calm, controlled layout.** Generous whitespace, orthogonal connectors,
   labels as small tinted chips. It should feel engineered, not auto-generated.

The footer reads "Made with FACETpy" in muted slate, bottom-right.

---

## 3. Canvas, background & title (fixed-width contract)

Every diagram uses a **fixed width** so they all scale identically when dropped
into a document — the title block then renders at the same size everywhere and
the image always fits the page.

- **Width is fixed at `CANVAS_W` (1000 units). Never change it.** Lay all content
  out within `x ∈ [0, 1000]`. Grow the diagram **vertically** (height) when you
  need more room — never widen it. If content feels tight horizontally, shrink
  node widths or stack vertically instead of exceeding 1000.
- Pass only the height: `Diagram(height, title=..., subtitle=...)`.
- **Background is transparent by default** — it drops cleanly onto any page.
  The grid, EEG wave, cards and every other feature still render (they sit on
  transparency). Leave `background=False`; only set `background=True` for a
  standalone image that wants the tinted canvas.
- **The title block is optional.** With a title, reserve `TITLE_MARGIN` (~90px)
  at the top and start nodes around `y≈100`. Without a title, start content near
  the top (`y≈40`) so there's no empty band.

## Layout discipline

- Give nodes room; crowding is the fastest way to lose the "engineered" feel.
  Add height rather than shrink gaps.
- Prefer **orthogonal** connector routing (horizontal/vertical segments). Route
  long edges *around* nodes, not through them — add polyline waypoints.
- **Stop edges a few px short of the target node** (≈2–4px). Nodes are drawn on
  top of edges, so a marker (arrowhead, diamond, triangle) whose endpoint lands
  exactly on a card border gets hidden beneath the card. End the connector just
  outside the border so the marker stays visible.
- Place edge labels at segment midpoints; multiplicities right next to the node
  they belong to.
- Align nodes to an implicit grid (multiples of ~10) so headers and columns line
  up. Members inside cards already align automatically via the toolkit.
- Size cards to their content. The toolkit computes class-box height from the
  row counts; for generic cards pick a height that leaves even padding.

---

## 4. Toolkit API (`assets/facetpy_svg.py`)

Import and compose. Every node function returns a **geometry dict** — pass it to
`Diagram.add(...)` and read anchor points from it (or via `side(box, name)`).

```python
from facetpy_svg import (Diagram, class_box, card, pill, capsule, decision,
                         container, note, lifeline, activation, message,
                         edge, side)

d = Diagram(height, title="...", subtitle="...")   # width is fixed (CANVAS_W=1000)
# background transparent by default; title optional (omit `title` to skip it)
box = class_box(x, y, w, name, attrs, methods)   # attrs/methods: (vis, name, type)
d.add(box, other_box, ...)                        # nodes (drawn on top)
d.add_edge(edge([p1, p2, ...], ...))              # connectors (drawn beneath)
d.save("out.svg")
d.render_png("out.png", svg_path="out.svg", width=1280)   # preview via rsvg-convert
```

**Geometry dict keys:** `x, y, w, h, cx, cy` and anchor tuples `top, bottom,
left, right`. `side(box, "center")` gives the centre.

### Node primitives
- `class_box(x, y, w, name, attrs, methods, stereotype=None)` — UML class.
  `attrs`/`methods` are lists of `(visibility, name, type)`; `visibility ∈
  {"+","-","#"}`; empty `type` string = omit. Height is automatic.
- `card(x, y, w, h, title, lines=None, stereotype=None)` — generic titled box
  for architecture / package / component nodes. `lines` = list of body strings.
  `h` is a **minimum**: the card grows to fit its lines so text never spills
  past the border. Read the real height from the returned geometry dict (`["h"]`)
  before placing anything beneath it.
- `pill(x, y, w, h, text, accent=None)` — flow **process** node (tinted, left
  accent bar). Rounds fully when `h < 40`.
- `capsule(x, y, w, h, text)` — flow **terminator** (start/end), gradient stadium.
- `decision(cx, cy, w, h, text)` — flow **decision** diamond (centre-anchored).
- `container(x, y, w, h, title)` — dashed grouping frame with a title tab
  (swimlanes, packages, subsystems). Add it **first** so nodes sit on top.
- `note(x, y, w, h, text)` — annotation with a folded corner.
- `lifeline(cx, top, bottom, name, w=140)`, `activation(cx, y1, y2)`,
  `message(x1, x2, y, text, dashed=False, ret=False)` — sequence diagrams.

### Connectors — `edge(points, marker_end="arrow", marker_start=None, label=None, mult=None)`
- `points`: list of `(x, y)`. Two points = straight; more = orthogonal route.
- markers: `"arrow"` (association/flow), `"arrow-open"`, `"diamond"` (composition),
  `"diamond-open"` (aggregation), `"triangle"` (generalization/inheritance), or
  `None`. UML convention: diamond/triangle go on the **owner/parent** end, so
  set them via `marker_start` and route that end to the parent.
- `label=(text, x, y)` draws a chip **centered on (x, y)** that auto-sizes to
  its text (guillemet stereotypes like `«uses»` included) — so give the segment
  midpoint as (x, y) and it won't clip. `mult=[(text, x, y), ...]` places
  multiplicities like `"1"`, `"*"`, `"0..1"` (left-anchored at each point).

---

## 5. Per-diagram recipes

### Class diagram
Use `class_box`. Relationships:
- **Association** → `edge(..., marker_end="arrow")`, with `mult` on both ends.
- **Composition** (owns) → `marker_start="diamond"` at the owner.
- **Aggregation** (has) → `marker_start="diamond-open"`.
- **Inheritance** (is-a) → `marker_start="triangle"` at the parent (child →
  parent, triangle points at the parent).
Stereotypes (`«interface»`, `«abstract»`) via the `stereotype=` arg.

### Flow / pipeline diagram
`capsule` for start/end, `pill` for steps, `decision` for branches. Label
decision exits (`yes`/`no`). Keep the main path on one vertical spine; branch
sideways and rejoin. A left-accent `pill` reads as a FACETpy Processor step.
Use `note` for regex/parameters/side comments.

### Architecture / package diagram
`container` for subsystems/layers, `card` for modules/components inside them
(mirror `src/facet/`: core, io, preprocessing, correction, evaluation, ...).
`card` `lines=` list key classes/responsibilities. Connect with labelled
`edge`s (`"depends on"`, `"uses"`). Draw containers before their cards.

### Sequence diagram
`lifeline` per participant across the top; `message` for calls (solid) and
returns (`ret=True`, dashed); `activation` bars over the busy span of a
lifeline. Keep participants evenly spaced; order messages top-to-bottom.
- **Combined fragments** (`loop`, `alt`, `opt`) → `container(...)` with the label
  as its title. A lifeline usually runs *through* the frame, and lifelines are
  drawn on top — so add the `container` frames **after** the lifelines (or draw
  the tab labels last) so the title tab stays legible instead of being sliced by
  a dashed lifeline. The frame fill is ~transparent, so drawing it over the
  lifelines barely dims them.
- Give the tab enough clearance: start the frame so its title tab doesn't sit
  exactly on a busy lifeline crossing, and keep the first message a row below the
  tab.
- With many participants (6–8), narrow the lifelines (`w≈115–128`) and give long
  names breathing room from the connectome dot; if a name still crowds the dot,
  shorten it or widen just that card.

---

## 6. Worked example (class diagram)

```python
from facetpy_svg import Diagram, class_box, edge, side
d = Diagram(660, title="Restaurant", subtitle="domain model")  # width fixed at 1000
cust  = class_box(70, 120, 220, "Customer",
                  [("-", "name", "str"), ("-", "phone", "str")],
                  [("+", "placeOrder()", "Order")])
order = class_box(70, 400, 230, "Order",
                  [("-", "id", "int"), ("-", "total", "float")],
                  [("+", "checkout()", "")])
d.add(cust, order)
mx = cust["cx"]
d.add_edge(edge([(mx, cust["y"]+cust["h"]+3), (mx, order["y"]-3)],  # stop short of cards
                label=("places", mx+8, 448),
                mult=[("1", mx+6, cust["y"]+cust["h"]+16),
                      ("*", mx+6, order["y"]-8)]))
d.render_png("restaurant.png", svg_path="restaurant.svg", width=1280)
```

---

## 7. Custom & creative elements

The node primitives cover the common cases; real diagrams sometimes need more.
Add whatever the content calls for — the goal is a *clear, expressive* diagram,
not one limited to five box types. Keep it unmistakably FACETpy by drawing from
the same materials the toolkit uses.

**The materials you build from** (all public in `facetpy_svg`):
- `C` — the active palette dict: `C["ink"]`, `C["blue"]`, `C["blue400"]`,
  `C["blue200"]`, `C["slate"]`, `C["surface"]`, `C["tint"]`, `C["navy_d"]`, …
  (theme-aware — reads light or dark automatically). Never hard-code a hex.
- `FONT` — the one type family; use it on every `<text>`.
- Shared defs by id: `url(#fp-shadow)`, `url(#fp-header)` (the navy→blue
  gradient), `url(#fp-grid)`, and markers `fp-arrow`, `fp-arrow-open`,
  `fp-diamond`, `fp-diamond-open`, `fp-triangle`.
- Public low-level helpers: `text(...)`, `node_dot(cx, cy)`,
  `rounded_top(x, y, w, head, r)`, `card_shell(x, y, w, h)`, `divider(x, y, w)`.
- `Diagram.add_defs(*snippets)` — register your OWN `<marker>` / gradient /
  `<symbol>` so custom shapes can use `url(#…)` like the built-ins. Prefix ids
  with `fp-` to signal they belong to the system.
- `Diagram.add(raw_svg_string)` / `add_edge(...)` — any string you pass is
  emitted verbatim, so hand-written SVG drops straight in on the right layer.

### Recipe: a custom marker + a representative glyph

```python
from facetpy_svg import Diagram, card, text, node_dot, C, FONT
d = Diagram(400, title="Custom bits", subtitle="demo")

# 1) A custom "data flow" arrow: register a filled arrowhead in brand blue.
d.add_defs(f'''<marker id="fp-flow" markerWidth="12" markerHeight="12"
    refX="9" refY="5" orient="auto">
    <path d="M1 1 L10 5 L1 9 Z" fill="{C['blue']}"/></marker>''')
d.add_edge(f'<path d="M120 200 H320" fill="none" stroke="{C["blue"]}" '
           f'stroke-width="2" stroke-dasharray="6 4" marker-end="url(#fp-flow)"/>')

# 2) A representative glyph — a tiny EEG signal chip built from tokens.
d.add(f'<rect x="120" y="120" width="90" height="46" rx="10" fill="{C["tint"]}" '
      f'stroke="{C["blue"]}" stroke-opacity="0.5"/>'
      f'<polyline points="130,143 145,143 152,126 160,160 168,143 200,143" '
      f'fill="none" stroke="{C["blue400"]}" stroke-width="2"/>')

# 3) Reuse helpers so it matches: branded text + the logo connectome dot.
d.add(text(120, 190, "raw EEG", size=12, fill=C["slate"]))
d.add(node_dot(120, 120))
```

Guidelines for custom work:
- Pull every colour from `C`, every label through `text(...)`/`FONT`, and lean on
  `url(#fp-shadow)` + `url(#fp-header)` so custom cards match the built-in ones.
- Respect the layers: connectors/background via `add_edge`, shapes/labels via
  `add`. Register defs before you reference them.
- Custom markers/gradients get an `fp-` id prefix. Keep stroke weights and corner
  radii close to the toolkit's (≈1.25–2px strokes, r≈10–12) so nothing clashes.
- If you invent something broadly reusable (a new node type, a common icon),
  say so when you return — it's a candidate to fold into the toolkit later.

## 8. Before you return — self-check

Render the PNG and look at it. Verify:
- [ ] **Occlusion pass — nothing is covered, obstructed, overlapped, or
      invisible.** Scan every element and confirm it is fully visible: no node
      sits on top of another, no label/text is hidden behind a card or another
      chip, no marker (arrowhead/diamond/triangle) is clipped under a node, no
      connector disappears into or behind a box, nothing runs off the canvas
      edge, and no member row is cut off by its card border. If any element is
      even partially obscured, move it, reroute the edge, resize the box, or grow
      the canvas height — then re-render and scan again.
- [ ] No overlaps: nodes don't touch, connectors don't cross nodes, labels are
      readable and not colliding with lines or the EEG wave.
- [ ] Every titled node has the gradient header + connectome dot.
- [ ] Grid + title EEG wave present; footer present.
- [ ] Edges use the correct UML markers and have multiplicities where relevant,
      and every marker is fully visible (not clipped under a card — end the edge
      a few px short of the node).
- [ ] Width is the fixed `CANVAS_W` (1000); all content sits within it. Add
      height, never width, if it feels cramped.
- [ ] Background is transparent (no `background=True`) unless asked otherwise.
- [ ] No large empty band at the top — with a title start ~y=100, without one ~y=40.
- [ ] Colours come from the tokens only; text is legible against its fill.

If anything is off, adjust coordinates and re-render before returning. Iterating
on the PNG is expected — it usually takes two or three passes to place edges
cleanly.
