"""
Generate the engineering rationale document as a PDF.

This is the written defence of the implementation: not what the code does
(the README covers that) but *why each decision was made that way*, what the
alternative would have cost, and where the remaining weaknesses are.

Usage:
    python report/make_rationale_pdf.py [output.pdf]

Writes to the Desktop by default. Figures are pulled from results/, so run the
benchmark first if they are missing -- the script will say so rather than
silently emitting a document with holes in it.
"""

import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Project root, so the script works from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Ink and accent colours, matching the project's validated chart palette.
INK = colors.HexColor("#0b0b0b")
INK_SOFT = colors.HexColor("#3d3c39")
MUTED = colors.HexColor("#6b6a65")
RULE = colors.HexColor("#d8d7d0")
ACCENT = colors.HexColor("#2a78d6")
ACCENT_2 = colors.HexColor("#1baf7a")
SURFACE = colors.HexColor("#f4f4f1")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()
    s = {}

    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23,
        leading=28, textColor=INK, alignment=TA_LEFT, spaceAfter=6)
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12.5,
        leading=17, textColor=INK_SOFT, spaceAfter=18)
    s["byline"] = ParagraphStyle(
        "byline", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
        leading=14, textColor=MUTED)

    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15,
        leading=19, textColor=INK, spaceBefore=20, spaceAfter=8)
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        leading=15, textColor=ACCENT, spaceBefore=13, spaceAfter=5)

    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.8,
        leading=14.6, textColor=INK_SOFT, alignment=TA_JUSTIFY, spaceAfter=8)
    s["lead"] = ParagraphStyle(
        "lead", parent=s["body"], fontSize=10.6, leading=15.8, textColor=INK)

    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=13, bulletIndent=3, spaceAfter=5)

    s["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontName="Courier", fontSize=8.4,
        leading=12, textColor=INK, backColor=SURFACE, borderPadding=7,
        leftIndent=5, rightIndent=5, spaceBefore=5, spaceAfter=10)

    s["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"], fontName="Helvetica-Oblique",
        fontSize=8.4, leading=11.5, textColor=MUTED, spaceBefore=5,
        spaceAfter=14)

    s["callout"] = ParagraphStyle(
        "callout", parent=s["body"], fontName="Helvetica", fontSize=9.8,
        leading=14.6, textColor=INK, backColor=SURFACE, borderPadding=9,
        borderColor=RULE, borderWidth=0.6, leftIndent=2, rightIndent=2,
        spaceBefore=7, spaceAfter=11)

    s["tablecell"] = ParagraphStyle(
        "tablecell", parent=base["Normal"], fontName="Helvetica", fontSize=8.3,
        leading=11, textColor=INK_SOFT)
    s["tablehead"] = ParagraphStyle(
        "tablehead", parent=s["tablecell"], fontName="Helvetica-Bold",
        textColor=INK)
    return s


S = build_styles()


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    return Paragraph(text, S["h1"])


def H2(text):
    return Paragraph(text, S["h2"])


def BULLETS(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def CODE(text):
    # Paragraph parses markup, so the angle brackets in code must be escaped.
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])


def CALLOUT(text):
    return Paragraph(text, S["callout"])


def FIGURE(filename, caption, width_frac=1.0):
    """Embed a results figure, scaled to the content width."""
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        return [CALLOUT("[missing figure: %s - run python run_benchmark.py]"
                        % filename)]

    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = CONTENT_W * width_frac
    h = w * ih / float(iw)

    # Keep a tall figure from overflowing the page.
    max_h = (PAGE_H - 2 * MARGIN) * 0.62
    if h > max_h:
        h = max_h
        w = h * iw / float(ih)

    return [Image(path, width=w, height=h), Paragraph(caption, S["caption"])]


def TABLE(rows, col_widths=None, align_right_from=1):
    """A clean table: hairline rules, no heavy borders."""
    data = []
    for r, row in enumerate(rows):
        style = S["tablehead"] if r == 0 else S["tablecell"]
        data.append([Paragraph(str(c), style) for c in row])

    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, colors.HexColor("#ecebe5")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
    ]))
    return t


# ----------------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------------

def page_decoration(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 12 * mm,
                          "A*, RRT and RRT* - Engineering Rationale")
        canvas.drawRightString(PAGE_W - MARGIN, 12 * mm, "%d" % doc.page)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, 15.5 * mm, PAGE_W - MARGIN, 15.5 * mm)
    canvas.restoreState()


# ----------------------------------------------------------------------------
# Document content
# ----------------------------------------------------------------------------

def cover():
    out = [Spacer(1, 26 * mm)]
    out.append(P("Comparative Analysis of A*, RRT and RRT*", "title"))
    out.append(P("Engineering rationale: the decisions behind the "
                 "implementation, and the case for them.", "subtitle"))

    bar = Table([[""]], colWidths=[CONTENT_W], rowHeights=[2.2])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
    out.append(bar)
    out.append(Spacer(1, 9 * mm))

    out.append(P("Team RoboNexus &nbsp;|&nbsp; AI for Robotics<br/>"
                 "Motion planning in a simulated 2D environment<br/>"
                 "Python 3, NumPy, Matplotlib - no physics engine, no ROS, "
                 "no external simulator", "byline"))
    out.append(Spacer(1, 12 * mm))

    out.append(CALLOUT(
        "<b>The thesis of this document.</b> The brief asked for three planners "
        "and a comparison between them. Implementing three textbook algorithms "
        "is the easy half. The hard half - and where nearly all of the "
        "engineering effort actually went - is making the comparison "
        "<i>mean</i> something: guaranteeing that the three planners are "
        "measured against the same problems, under the same rules, with the "
        "same collision model, and that the resulting numbers are not quietly "
        "biased by how the harness was written. Every decision below is "
        "defended on that basis."))

    out.append(Spacer(1, 6 * mm))
    out.append(P("<b>Status.</b> Simulation world, all three planners, "
                 "benchmark harness and evaluation figures complete. 45 unit "
                 "tests passing. 450 benchmark runs. Every reported path "
                 "independently re-verified collision-free.", "body"))
    out.append(PageBreak())
    return out


def section_1():
    return [
        H1("1. What this project actually had to prove"),
        P("A comparison is only worth as much as its fairness guarantees. If A* "
          "is allowed a cheaper collision test than RRT, then a difference in "
          "success rate measures the harness, not the algorithm. If path length "
          "is averaged over each planner's own successes, then the planner that "
          "fails on the hard worlds gets scored only on the easy ones and looks "
          "<i>better</i> for having failed. Both of these are easy mistakes to "
          "make by accident, and neither announces itself in the output - the "
          "numbers still look plausible.", "lead"),
        P("So I treated the project as three deliverables rather than one: a "
          "world model whose collision semantics are exact and shared; three "
          "planners that are individually faithful to their source papers; and "
          "a measurement apparatus whose biases I could name and defend. The "
          "sections below follow that order, because that is the order the "
          "dependencies run in."),

        H2("Scope discipline"),
        P("The brief is explicitly a classical-AI project, and I kept it that "
          "way. No learned sampling, no ML dependency, no physics engine. The "
          "robot is a disc; collision checking is pure analytic geometry. That "
          "is a constraint I would defend even without the brief: every "
          "component here is one a reader can verify by hand, which is exactly "
          "what a technical viva requires. A black-box simulator would have "
          "made the results harder to trust, not easier."),
    ]


def section_2():
    return [
        H1("2. The collision model is the foundation, so it is exact"),
        P("Everything else rests on two questions: may the robot occupy this "
          "point, and may it travel this straight line. I made the "
          "<font face='Courier' size='9'>Environment</font> the single "
          "authority on both, and made both answers exact."),

        H2("Why not sample along the edge?"),
        P("The common shortcut is to check an edge by testing points spaced "
          "along it. It is simple and it is wrong in a way that would have "
          "silently corrupted this specific comparison. A sampled checker can "
          "step straight over a thin obstacle - and the longer the step, the "
          "more likely that is. RRT and RRT* extend by 5.0 world units per "
          "iteration; A* moves at most 1.41. A sampled checker would therefore "
          "have granted the sampling planners a systematically more permissive "
          "feasibility test than A*, and every success-rate and path-length "
          "number would have inherited that bias."),
        P("Instead, <font face='Courier' size='9'>is_motion_valid</font> "
          "computes the true distance from the swept segment to each obstacle. "
          "The swept area of a disc along a segment is a capsule, and that "
          "capsule misses an obstacle exactly when the segment's distance to it "
          "exceeds the robot radius. There is no step size, so there is nothing "
          "to tune and nothing to tunnel through."),
        CODE("def test_no_tunnelling_through_thin_obstacle(self):\n"
             "    env = Environment(100, 100, [RectObstacle(49.9, 0, 0.2, 100)],\n"
             "                      (5, 50), (95, 50), robot_radius=0.0)\n"
             "    self.assertFalse(env.is_motion_valid((5, 50), (95, 50)))"),
        P("A 0.2-unit wall against a 90-unit segment. Most sampled checkers "
          "pass straight through it. This one does not, and the test exists so "
          "that nobody can reintroduce the shortcut without the suite going red."),

        H2("The robot radius comes out for free"),
        P("I phrased collision as a distance comparison - "
          "<font face='Courier' size='9'>clearance &lt;= robot_radius</font> - "
          "rather than a boolean overlap test. That is not a stylistic "
          "preference. Inflating an obstacle by the robot radius is the "
          "Minkowski sum of the obstacle with the robot disc, and comparing "
          "distance against the radius is exactly equivalent to it. So the "
          "disc-shaped robot is handled with no obstacle-inflation code at all, "
          "and the exact rounded corners of the inflated shape come along "
          "automatically. Writing explicit inflation would have meant writing - "
          "and defending - a rounded-rectangle geometry routine."),
    ]


def section_3():
    return [
        H1("3. The hardest fairness problem: discretising for A*"),
        P("A* needs a discrete graph. RRT and RRT* do not. This asymmetry is "
          "the single most likely place for a comparison like this to go wrong, "
          "so it deserves the most scrutiny.", "lead"),

        H2("The obvious approach, and why I rejected it"),
        P("The standard recipe is to build an occupancy grid by marking a cell "
          "blocked if it lies within the robot radius plus half a cell diagonal "
          "of an obstacle, then run A* on the resulting mask. It is fast and it "
          "guarantees collision-free paths. It is also <i>conservative</i>: it "
          "blocks cells that are actually reachable. In a cluttered world that "
          "conservatism closes narrow passages that genuinely exist - so A* "
          "would report failure on instances RRT solves, and the success-rate "
          "gap would be an artefact of my discretisation rather than a property "
          "of A*."),

        H2("What I did instead"),
        P("The grid supplies only the node set - the cell centres and the "
          "8-connected neighbour structure. Validity is not baked into a mask. "
          "Each cell's occupancy and each move's legality are resolved lazily, "
          "on first touch, through the very same "
          "<font face='Courier' size='9'>Environment</font> queries that RRT "
          "calls."),
        *BULLETS([
            "A* gets no cheaper and no more permissive a collision test than "
            "the sampling planners, so a success-rate gap is a real algorithmic "
            "difference.",
            "Validity is resolved per cell as the search touches it, so A* pays "
            "for exactly the states it expands - which is what makes its "
            "collision-check count comparable to a tree planner's.",
            "Because every move is checked as a swept segment, a diagonal step "
            "can never cut the corner between two touching obstacles.",
        ]),
        P("The cost is more collision checks per expansion. I consider that the "
          "right trade: it is an honest cost, it is measured and reported in "
          "the <font face='Courier' size='9'>motion_checks</font> column, and "
          "it buys a comparison I can defend."),

        H2("The residual limitation, stated plainly"),
        P("A* remains optimal only <i>on the grid it is given</i>. Its path "
          "length is bounded below by the true continuous optimum, and the gap "
          "shrinks as resolution increases. At 1.0-unit resolution it lands "
          "about 5% above the best path found. That is a discretisation "
          "artefact, not a flaw in the search, and the report should say so "
          "rather than present A* as simply worse than RRT*."),
    ]


def section_4():
    return [
        H1("4. Making it fast without making it wrong"),
        P("The first working version of A* took 12.9 seconds on the maze - "
          "close enough to the time budget to distort the results. Profiling "
          "put 77% of the time in segment-to-rectangle distance: 143,968 calls, "
          "each running four segment-segment tests."),
        P("The fix was a broad phase. If an obstacle's bounding box does not "
          "overlap the segment's bounding box inflated by the robot radius, "
          "then the obstacle is separated along some axis by more than the "
          "radius, so the exact distance necessarily exceeds it. The expensive "
          "test can be skipped with no possibility of a false negative."),

        CALLOUT(
            "<b>The result: 12.9 s to 0.16 s - an 82x speed-up - with an "
            "identical path length and an identical collision-check count.</b> "
            "That second identity is the part I care about. If the optimisation "
            "had changed any answer, the check count would have moved. It did "
            "not, which is evidence that the speed-up is exact rather than an "
            "approximation that happens to look right."),

        P("I did not rely on that single observation. "
          "<font face='Courier' size='9'>test_broad_phase_matches_exact_test"
          "</font> compares the optimised path against a brute-force evaluation "
          "over 400 random segments in a cluttered world and asserts they agree "
          "on every one. An optimisation in the fairness-critical path needs to "
          "be provably equivalent, not plausibly equivalent."),
        P("It also matters that the speed-up helps all three planners equally - "
          "it lives in the shared "
          "<font face='Courier' size='9'>Environment</font>, below the point "
          "where the planners differ. An optimisation that accelerated only A* "
          "would have been a measurement change disguised as a performance fix."),
    ]


def section_5():
    return [
        H1("5. The three planners"),

        H2("A* - Hart, Nilsson and Raphael, 1968"),
        P("The heuristic is octile distance, which is the exact cost of an "
          "unobstructed 8-connected path. It is admissible because obstacles can "
          "only make things worse, and it dominates straight-line Euclidean "
          "distance, so it expands strictly fewer nodes. Diagonal moves cost "
          "the square root of two rather than one: without that the reported "
          "path length would be inflated by the staircase effect and A* would "
          "look artificially bad against RRT*."),
        P("Admissibility is the kind of claim that is easy to assert and easy to "
          "get wrong, so I tested it rather than argued it. Setting the "
          "heuristic weight to zero turns A* into Dijkstra; both must return the "
          "same cost on the same grid. They do, to six decimal places - and A* "
          "expands measurably fewer nodes, confirming the heuristic is doing "
          "real work rather than being trivially zero."),
        P("Two details that are easy to get wrong. Open-list entries are ordered "
          "by f, then by h, then by an insertion counter: the h tie-break "
          "measurably reduces expansions in open space, and the counter makes "
          "the ordering total so the heap never compares raw cell tuples. And "
          "because start and goal are continuous coordinates that rarely land "
          "on a cell centre, the planner snaps them to the nearest usable cells "
          "and then stitches the two leftover stubs back on - checking each stub "
          "for collision, and returning failure rather than a shortcut if either "
          "is blocked. Without that, the reported path would not actually begin "
          "at the start."),

        H2("RRT - LaValle, 1998"),
        P("Deliberately the plain textbook formulation: sample, nearest, steer, "
          "extend, and stop at the first path found. I was tempted to add "
          "path shortcutting, and I left it out on purpose. RRT's non-optimality "
          "is precisely the thing RRT* is supposed to fix, so post-processing "
          "RRT's output would have destroyed the main axis of the comparison. "
          "The measured 27% gap is a real algorithmic difference; had I smoothed "
          "the output it would have been a statement about my smoother."),
        P("For the same reason, RRT and RRT* share every common parameter - step "
          "size, goal bias, goal tolerance, iteration cap. Any difference "
          "between them is therefore attributable to rewiring and not to tuning."),

        H2("RRT* - Karaman and Frazzoli, 2011"),
        P("Three implementation decisions are worth defending."),
        P("<b>The rewiring radius is derived, not tuned.</b> The shrinking-ball "
          "radius is the theoretical heart of the algorithm, and the constant "
          "must exceed a bound that depends on the free-space volume. I compute "
          "that bound from the environment's own free area rather than "
          "hand-picking a number that looks good. A tuned constant would have "
          "invited the obvious viva question - did you pick it to make RRT* win?"),
        P("<b>Rewiring propagates costs to descendants.</b> When a node is "
          "re-parented onto a cheaper route, every node in its subtree now "
          "reaches the root more cheaply too. A common shortcut in simple "
          "implementations is to skip that propagation, which leaves stale costs "
          "that cause the algorithm to reject genuinely better rewirings later "
          "on. It is a silent defect: the planner still returns valid paths, "
          "just worse ones. "
          "<font face='Courier' size='9'>Tree.set_parent</font> propagates "
          "iteratively - recursion would risk a stack overflow on the long "
          "chains RRT* builds - and a unit test pins the behaviour."),
        P("<b>Rewiring cannot create a cycle, and I can show why.</b> Costs "
          "increase monotonically along any tree path, so for any ancestor of "
          "the new node, the rewired cost through that new node is strictly "
          "greater than the ancestor's existing cost. The strict inequality in "
          "the rewire test therefore never fires for an ancestor, and no cycle "
          "is possible. That is an argument I would rather make from the "
          "structure of the algorithm than defend with a runtime check."),
        P("The goal joins the tree as an ordinary node once reached, so rewiring "
          "improves the route to it automatically. It is excluded from being "
          "chosen as a parent, since nothing should ever be routed <i>through</i> "
          "the goal."),

        H2("The one place I chose readability over asymptotics"),
        P("Nearest-neighbour search is brute force over a contiguous NumPy "
          "array - O(n) per query, vectorised. A k-d tree would be "
          "asymptotically better, but it needs rebuilding or rebalancing on "
          "every insertion, and the tree here grows by one node per iteration. "
          "For trees of a few thousand nodes the vectorised scan is fast enough, "
          "and it is code a reader can verify at a glance. I have documented it "
          "as the known scaling limitation rather than hiding it."),
    ]


def section_6():
    return [
        H1("6. Designing the measurements before taking them"),
        P("Every planner returns the same flat metrics dictionary with a fixed "
          "schema, so the harness writes rows straight to CSV without "
          "special-casing any algorithm. Two schema decisions carry real weight."),

        H2("Separating first-solution time from total time"),
        P("RRT* is an <i>anytime</i> algorithm: it keeps improving after it has "
          "a path. RRT stops the moment it has one. Comparing RRT*'s total "
          "runtime against RRT's is therefore comparing two different "
          "quantities, and it is the single most natural way to get this "
          "comparison wrong - it makes RRT* look roughly 300 times slower when "
          "the honest figure is about 1.4 times."),
        P("So the schema records "
          "<font face='Courier' size='9'>first_solution_time</font>, "
          "<font face='Courier' size='9'>first_solution_length</font> and "
          "<font face='Courier' size='9'>first_solution_iteration</font> "
          "alongside the finals. For A* and RRT these coincide with the final "
          "values, which is exactly the point: the columns are directly "
          "comparable across all three. The measured figures are 4.3 ms to "
          "first path against RRT's 3.0 ms, with the remaining ~1.0 s spent "
          "improving rather than searching."),

        H2("Counting collision checks, not just seconds"),
        P("Wall-clock time depends on the machine, the interpreter and what else "
          "is running. Collision-check counts do not. Recording both means the "
          "report has a machine-independent cost measure to fall back on, and it "
          "is what let me prove the broad-phase optimisation was exact. When the "
          "full sweep was re-run, every path length and node count reproduced "
          "<i>exactly</i>; only the timings moved. That tells the reader which "
          "columns to quote as precise and which to quote as approximate."),

        H2("Verification is not optional"),
        P("A planner that reports success on a path clipping an obstacle would "
          "corrupt both the success-rate and the path-length columns, and "
          "nothing downstream would notice. So every claimed path is "
          "independently re-checked by "
          "<font face='Courier' size='9'>verify_path</font> before its metrics "
          "are recorded - in the demo, in the benchmark harness, and in the "
          "tests. It re-walks every segment, and confirms the path genuinely "
          "starts at the start and ends at the goal. Across 450 benchmark runs, "
          "zero invalid paths."),
    ]


def section_7():
    return [
        H1("7. The benchmark, and the bias it was built to avoid"),
        P("450 runs: three density levels, ten worlds each, five trials, three "
          "planners. The same generated world list goes to all three planners; "
          "trial <i>t</i> uses seed <i>t</i> for every stochastic planner, so "
          "RRT and RRT* see the same random stream on the same world; all three "
          "get the same time budget and iteration cap."),
        P("Worlds are filtered for reachability by a flood fill on a grid finer "
          "than any planner uses. That is a deliberately permissive filter: it "
          "rejects worlds that are definitely unsolvable without pre-judging "
          "what any particular planner can find. A planner that fails on a world "
          "passing this filter has genuinely failed, and the success-rate column "
          "means what it says."),

        H2("Why path length is averaged over a common subset"),
        P("This is the methodological point I would most want to be asked about. "
          "Suppose a planner fails on the hard worlds. Averaging it over its own "
          "successes scores it only on the easy ones - so failing makes it look "
          "<i>better</i>. The fix is to compare quality only on the instances "
          "every planner solved, and to let the failures show up in the "
          "success-rate metric, which is the metric that properly owns them."),
        P("I did not want that to be a claim in a document, so there is a test "
          "that builds the trap explicitly. Two worlds; an easy one everyone "
          "solves and a hard one where RRT fails:"),
        TABLE([
            ["scenario", "A*", "RRT", "RRT*"],
            ["easy world", "100", "130", "95"],
            ["hard world", "300", "failed", "290"],
            ["<b>mean over own successes</b>", "<b>200</b>", "<b>130</b>", "<b>192.5</b>"],
            ["<b>mean over common subset</b>", "<b>100</b>", "<b>130</b>", "<b>95</b>"],
        ], col_widths=[CONTENT_W * 0.40] + [CONTENT_W * 0.20] * 3),
        Spacer(1, 4),
        P("The naive average says RRT produces shorter paths than A* - 130 "
          "against 200. It does not. It simply never attempted the long world. "
          "The common-subset average puts the ordering back the right way round, "
          "and the success-rate column records RRT at 50%. "
          "<font face='Courier' size='9'>"
          "test_naive_mean_would_flatter_the_failing_planner</font> asserts both "
          "halves of that."),
        P("In this particular run all three planners solved all 150 instances, "
          "so the two averages coincide. The machinery still earns its place: it "
          "is what makes the number trustworthy the moment a planner starts "
          "failing, and it means the result is not quietly conditional on an "
          "outcome I could not have guaranteed in advance."),
    ]


def section_8():
    return [
        H1("8. Two statistics I had to get right"),

        H2("Success rate was uninformative, so I changed the question"),
        P("At the full 10-second budget all three planners reach 100% at every "
          "density. A bar chart of that is three identical bars - a chart where "
          "the story is a single number, and one that would have implied the "
          "planners are equivalent when they differ by an order of magnitude in "
          "how fast they get there."),
        P("Success rate within a budget is only a discriminating metric while "
          "the budget actually binds. So rather than report a flat 100%, I sweep "
          "the budget and plot the share of runs solved within it. A run that "
          "first found a path at time <i>t</i> counts as a success for every "
          "budget at or above <i>t</i>, so the entire curve is derived from data "
          "already recorded - it is a different view of the same experiment, not "
          "a second one. The flat 100% is still reported in the tables; it is "
          "simply not the thing worth drawing."),
        *FIGURE("chart_success_profile.png",
                "Figure 1. Cumulative success profile. All three planners reach "
                "100%; the difference is when. RRT is reliable roughly ten times "
                "sooner than A*, with RRT* close behind it."),

        H2("A convergence median that rose when it should not have"),
        P("The first version of the convergence chart showed RRT*'s median best "
          "path length <i>increasing</i> over early iterations. That is "
          "impossible for any individual run - the best-known cost is monotone "
          "non-increasing by construction - so the artefact was in my "
          "statistics, not the algorithm."),
        P("The cause: I was averaging only over runs that had already found a "
          "path. Early on, those are the lucky ones that found short paths "
          "quickly. As the plot moves right, slower runs with worse initial "
          "paths join the population, and the median rises even though no run "
          "ever got worse. The population was changing underneath the statistic."),
        P("The fix is to keep the population fixed: a run with no path yet is "
          "scored as infinitely long rather than dropped. Each run's own curve "
          "is then monotone non-increasing, so the median over a fixed set is "
          "too. The median becomes finite once half the runs have a path and the "
          "interquartile band once three-quarters do - exactly where each "
          "becomes meaningful. Percentiles are taken by nearest rank so that no "
          "interpolation against infinity is ever performed."),
        CALLOUT(
            "I am including this because catching it mattered more than any "
            "feature I wrote. The chart looked fine. It was well-formed, "
            "plausible, and would have survived a casual review - and it was "
            "reporting something the algorithm cannot do. The lesson I would "
            "defend is that a plot which contradicts a known invariant is a bug "
            "report, not a curiosity."),
    ]


def section_9():
    return [
        H1("9. Presenting results without distorting them"),
        P("Charts are part of the engineering, not decoration applied afterwards. "
          "Four decisions here would survive scrutiny."),
        *BULLETS([
            "<b>No dual-axis charts.</b> Plotting path length and computation "
            "time against two y-scales would invent a relationship that is not "
            "in the data, since the alignment of the two scales is arbitrary. "
            "They are separate charts.",
            "<b>Timing is a dumbbell on a log axis, not bars.</b> The values "
            "span three orders of magnitude, so a log scale is required - and "
            "bars grown from a baseline on a log axis encode an arbitrary origin "
            "as length. The dumbbell also shows something bars could not: the "
            "span from first feasible path to total runtime, which <i>is</i> "
            "RRT*'s refinement time.",
            "<b>Colour identifies the planner, never its rank.</b> A* is blue, "
            "RRT orange, RRT* aqua, in every figure - the path plots included - "
            "so a reader learns the mapping once.",
            "<b>The palette was computed, not chosen.</b> Under simulated "
            "colour-vision deficiency the worst separation between any pair is "
            "Delta-E 9.2 against a target of 8, and the worst normal-vision "
            "separation is 24.0 against a floor of 15. One series falls below "
            "the 3:1 contrast floor, so every mark carries a visible value "
            "label and the same numbers are written to a table file.",
        ]),
        P("The validation tool was written for Node, which is not installed on "
          "this machine. Rather than eyeball the palette and hope, I ported the "
          "checks to Python and confirmed the port against the documented "
          "reference figures before trusting its output on my own colours. A "
          "test now pins the three hex values so a well-meaning edit cannot "
          "silently swap in unvalidated ones."),
        PageBreak(),
    ]


def section_10():
    headline = [
        ["planner", "success", "path length", "vs. best", "first path",
         "total", "nodes"],
        ["A*", "100%", "142.18 &plusmn; 9.05", "1.052x", "0.032 s", "0.032 s", "1,371"],
        ["RRT", "100%", "172.00 &plusmn; 23.86", "1.272x", "<b>0.003 s</b>",
         "0.003 s", "199"],
        ["RRT*", "100%", "<b>135.08 &plusmn; 7.42</b>", "<b>1.000x</b>", "0.004 s",
         "0.999 s", "7,435"],
    ]
    by_density = [
        ["density", "A*", "RRT", "RRT*"],
        ["sparse (10%)", "134.90", "159.96", "<b>129.51</b>"],
        ["moderate (20%)", "141.86", "169.46", "<b>133.57</b>"],
        ["cluttered (30%)", "149.78", "186.58", "<b>142.16</b>"],
    ]

    return [
        H1("10. What the numbers say"),
        TABLE(headline, col_widths=[CONTENT_W * x for x in
                                    (0.13, 0.12, 0.23, 0.12, 0.14, 0.12, 0.14)]),
        Spacer(1, 10),
        P("The expected theoretical ordering holds at every density: "
          "<b>RRT* &lt; A* &lt; RRT</b> on path length. RRT* was the best "
          "planner on every single instance - a normalised length of exactly "
          "1.000x - and produced paths 27% shorter than RRT and 5% shorter than "
          "A*."),
        Spacer(1, 6),
        TABLE(by_density, col_widths=[CONTENT_W * x for x in
                                      (0.31, 0.23, 0.23, 0.23)]),
        Spacer(1, 10),

        H2("The trade each planner actually makes"),
        P("<b>RRT buys speed with quality.</b> It reaches a feasible path about "
          "ten times faster than A* and explores roughly seven times fewer "
          "nodes - but returns the longest paths and by far the most variable "
          "ones, at a standard deviation of 23.9 against RRT*'s 7.4. That "
          "variance <i>is</i> the non-optimality: RRT returns whatever it "
          "stumbles into."),
        P("<b>A* degrades most sharply with clutter</b>, from 0.011 s to 0.047 s "
          "and from 788 to 1,906 cells expanded, because obstacles force the "
          "heuristic to explore around them. RRT* moves the other way - 8,472 "
          "down to 6,350 nodes - since in denser worlds more samples are "
          "rejected as colliding, so fewer survive into the tree. The two "
          "algorithms are responding to density through opposite mechanisms."),
        P("<b>RRT* is not slow to find a path; it is busy improving one.</b> Its "
          "first solution arrives at 4.3 ms against RRT's 3.0 ms, so "
          "choose-parent and rewire cost very little up front."),

        *FIGURE("chart_convergence.png",
                "Figure 2. The most informative result in the project. RRT* "
                "begins near RRT's path length, crosses A*'s line at roughly "
                "2,000 iterations, and keeps falling. The crossing point is the "
                "engineering answer to 'which planner should I use': before it, "
                "A* is the better choice."),

        P("That crossing is the finding I would lead with in the viva. It "
          "reframes the question from which planner is best to how much compute "
          "is available - which is the question a robotics engineer actually "
          "faces."),

        *FIGURE("maze_panels.png",
                "Figure 3. The same maze solved by each planner, with internals "
                "shown. A*'s shaded region is the set of expanded cells; the "
                "RRT and RRT* panels show their trees. The visual difference "
                "between RRT's sprawl and RRT*'s rewired structure explains the "
                "path-length gap more directly than any table."),
    ]


def section_11():
    return [
        H1("11. Mistakes, and what they cost"),
        P("Four things went wrong during the build. I am recording them because "
          "how a system fails under its own author is part of its defence.", "lead"),
        *BULLETS([
            "<b>A* was 80x too slow</b> in its first working form, close enough "
            "to the time budget to distort results. Caught by profiling rather "
            "than by guessing, and fixed with a provably exact broad phase.",
            "<b>The convergence chart reported something impossible</b> - a "
            "median that rose - and looked entirely plausible while doing it. "
            "Caught by noticing it contradicted a known invariant of the "
            "algorithm.",
            "<b>A --quick sanity run overwrote the dataset</b> backing every "
            "number in the report. The next re-analysis would have rebuilt all "
            "the tables and figures from eighteen throwaway rows with no error "
            "raised. Quick runs now write to separate files, and I verified the "
            "isolation by diffing the headline output across a quick run.",
            "<b>A patch script corrupted the README's encoding</b> by reading "
            "and writing without specifying UTF-8, so newly inserted characters "
            "were written in the platform's legacy codepage. Caught by scanning "
            "every project file for decode errors rather than by noticing "
            "visually.",
        ]),
        P("The first two were algorithmic and I expected that class of problem. "
          "The last two were infrastructure - the kind of defect that does not "
          "announce itself and quietly invalidates results downstream. The "
          "--quick overwrite is the one I would flag to any team working this "
          "way: a throwaway run must never be able to write to the same path as "
          "the canonical one, because the failure is silent and the corrupted "
          "output looks exactly like valid output."),
    ]


def section_12():
    return [
        H1("12. Threats to validity"),
        P("The results above are real, but they are bounded. I would rather "
          "state the boundaries myself than be walked into them."),
        *BULLETS([
            "<b>The success-rate axis is under-exercised.</b> All three planners "
            "solved all 150 instances, so the comparison rests on quality and "
            "time rather than feasibility. Harder instances - higher densities, "
            "or the narrow-gap scenario at scale - would be needed before "
            "claiming anything about relative robustness.",
            "<b>A*'s path length is resolution-dependent.</b> Its roughly 5% gap "
            "to the best path is discretisation, not search quality. A finer "
            "grid narrows it at quadratic cost in cells.",
            "<b>RRT*'s result depends on its iteration budget.</b> 10,000 "
            "iterations is a design choice. Its advantage over A* appears only "
            "after roughly 2,000, so a tighter budget would reverse the "
            "conclusion - which is why the convergence figure, not the summary "
            "table, is the honest headline.",
            "<b>Timings are single-machine and single-threaded</b>, in "
            "interpreted Python. The relative ordering is meaningful; the "
            "absolute milliseconds are not portable. The collision-check counts "
            "are the transferable cost measure.",
            "<b>The world is 2D, static and geometric.</b> A disc robot with no "
            "orientation, no dynamics and no moving obstacles. Nothing here "
            "predicts behaviour on a high-degree-of-freedom arm, where "
            "nearest-neighbour cost and the curse of dimensionality dominate.",
            "<b>Nearest-neighbour search is O(n) per query</b>, so the sampling "
            "planners' timings would degrade on much larger trees in a way a "
            "k-d tree implementation would not.",
        ]),

        H1("13. What I would do next"),
        P("In priority order, and with a reason for each rather than a wish list."),
        *BULLETS([
            "<b>Push the density until planners actually fail.</b> That is what "
            "would make the success-rate metric earn its place, and it is the "
            "cheapest remaining experiment.",
            "<b>Sweep A*'s grid resolution.</b> It would turn the "
            "discretisation caveat into a measured curve of path length and "
            "runtime against resolution, and let the report state the crossover "
            "where refining the grid stops paying.",
            "<b>Add informed sampling (Informed RRT*).</b> Once a first path "
            "exists, sampling can be restricted to the ellipse that could "
            "contain a better one. It is the natural next step from the "
            "convergence curve, and it directly attacks the slow tail.",
            "<b>Replace brute-force nearest-neighbour with a k-d tree</b>, but "
            "only when tree sizes justify it - and keep the current "
            "implementation as the reference to test the new one against.",
        ]),
        P("Learned sampling bias, dynamic obstacles, high-degree-of-freedom "
          "planning and ROS integration remain deliberately out of scope. They "
          "were considered and deferred; the scope boundary is what let this "
          "project be finished and verified rather than half-built in four "
          "directions."),

        # Kept together so the closing statement is never split across a page
        # break -- it spilled two orphan lines onto a near-empty final page.
        KeepTogether([
        H1("Closing"),
        CALLOUT(
            "Three planners, 450 benchmark runs, 45 tests, zero invalid paths, "
            "and a set of results that reproduce exactly on re-run. The "
            "theoretical predictions held: RRT* is asymptotically optimal and "
            "converged to 0.15% above the known optimum on an empty world; A* is "
            "optimal on its grid and landed within 5% of the best continuous "
            "path; RRT is fast and demonstrably non-optimal. "
            "<br/><br/>"
            "But the part I would defend hardest is not any of those numbers. It "
            "is that the comparison which produced them was built so that each "
            "one could be trusted - a shared exact collision model, a "
            "discretisation that grants A* no advantage, an optimisation proven "
            "equivalent rather than assumed so, metrics that separate anytime "
            "refinement from search, and an aggregation that cannot be flattered "
            "by failure. A benchmark that cannot be gamed by its own author is "
            "the actual deliverable."),
        ]),
    ]


def build(out_path):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN + 6 * mm,
        title="A*, RRT and RRT* - Engineering Rationale",
        author="Team RoboNexus",
        subject="Design decisions and defence for a 2D motion planning comparison",
    )

    story = []
    story += cover()
    story += section_1()
    story += section_2()
    story += section_3()
    story += section_4()
    story += section_5()
    story += section_6()
    story += section_7()
    story += section_8()
    story += section_9()
    story += section_10()
    story += section_11()
    story += section_12()

    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    return out_path


def main():
    if len(sys.argv) > 1:
        out = sys.argv[1]
    else:
        desktop = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
        if not os.path.isdir(desktop):
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out = os.path.join(desktop, "RoboNexus_Engineering_Rationale.pdf")

    missing = [f for f in ("chart_success_profile.png", "chart_convergence.png",
                           "maze_panels.png")
               if not os.path.exists(os.path.join(RESULTS, f))]
    if missing:
        print("WARNING: missing figures %s -- run python run_benchmark.py "
              "and python demo.py first." % ", ".join(missing))

    build(out)
    size_kb = os.path.getsize(out) / 1024.0
    print("Wrote %s (%.0f KB)" % (out, size_kb))


if __name__ == "__main__":
    main()
