"""Manim visualisation of the disease-direction projection.

Renders the same arithmetic as toy_projection_demo.py, in 2-D so the geometry is
readable. Text (pango) is used throughout rather than MathTex, so no LaTeX
package set is required.

    manim -ql manim_projection.py DiseaseProjection      # fast preview
    manim -qh manim_projection.py DiseaseProjection      # 1080p
"""
import numpy as np
from manim import (
    BLUE, DOWN, DashedLine, Arrow, Create, FadeIn, FadeOut, GREEN, GREY_B,
    LEFT, ORANGE, RIGHT, Scene, Text, UP, VGroup, Write, WHITE, YELLOW, RED,
    NumberPlane, ORIGIN, Transform, Angle, PI, Dot, SurroundingRectangle,
)

# Manim's default frame is x in [-7.11, 7.11], y in [-4, 4]. The vectors are
# drawn in a panel on the LEFT so the running score readout has the right half
# to itself; without the shift, paper B's tip and paper A's label both leave
# the frame.
S = 0.95                     # world units per vector unit
OX, OY = -3.4, -0.4          # where the vector origin sits on screen


def v2p(v):
    """vector -> manim point"""
    return np.array([v[0] * S + OX, v[1] * S + OY, 0.0])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


d = unit([1, 1])             # disease direction
h = unit([1, -1])            # probe: "recruited from hospital"
A = 3 * d + 1 * h            # up-camp paper
B = 3 * d - 1 * h            # down-camp paper
P = np.eye(2) - np.outer(d, d)
Ap, Bp = P @ A, P @ B


def cos(a, b):
    return float(unit(a) @ unit(b))


class DiseaseProjection(Scene):
    def construct(self):
        title = Text("Removing the disease direction", font_size=34).to_edge(UP)
        self.play(Write(title))

        plane = NumberPlane(
            x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1],
            x_length=3.5 * 2 * S, y_length=3.5 * 2 * S,
            background_line_style={"stroke_opacity": 0.18},
        ).move_to(np.array([OX, OY, 0.0]))
        self.play(Create(plane), run_time=1.2)

        # --- the two named directions -----------------------------------
        d_arrow = Arrow(v2p([0, 0]), v2p(d * 2.2), buff=0, color=GREY_B,
                        stroke_width=6)
        d_lbl = Text("disease direction", font_size=22, color=GREY_B)
        d_lbl.next_to(d_arrow.get_end(), UP, buff=0.10)

        h_arrow = Arrow(v2p([0, 0]), v2p(h * 2.2), buff=0, color=BLUE,
                        stroke_width=6)
        h_lbl = Text('probe: "recruited from hospital"', font_size=22,
                     color=BLUE)
        h_lbl.next_to(h_arrow.get_end(), DOWN, buff=0.10)

        self.play(Create(d_arrow), FadeIn(d_lbl))
        self.play(Create(h_arrow), FadeIn(h_lbl))
        self.wait(0.4)

        note = Text("neither is an axis — both are diagonals",
                    font_size=22, color=GREY_B).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.play(FadeOut(note))

        # --- the two papers ---------------------------------------------
        a_arrow = Arrow(v2p([0, 0]), v2p(A), buff=0, color=YELLOW, stroke_width=7)
        b_arrow = Arrow(v2p([0, 0]), v2p(B), buff=0, color=ORANGE, stroke_width=7)
        a_lbl = Text("paper A — ENRICHED", font_size=20, color=YELLOW)
        b_lbl = Text("paper B — DEPLETED", font_size=20, color=ORANGE)
        a_lbl.next_to(a_arrow.get_end(), DOWN + RIGHT, buff=0.10)
        b_lbl.next_to(b_arrow.get_end(), UP, buff=0.12)

        self.play(Create(a_arrow), FadeIn(a_lbl))
        self.play(Create(b_arrow), FadeIn(b_lbl))

        both = Text("Both are dominated by disease content —\n"
                    "they point almost the same way",
                    font_size=24, color=WHITE).to_edge(DOWN)
        self.play(FadeIn(both))
        self.wait(1.6)
        self.play(FadeOut(both))

        # --- scores before ----------------------------------------------
        box = VGroup(
            Text(f"cos(A, probe) = {cos(A, h):+.3f}", font_size=24,
                 color=YELLOW),
            Text(f"cos(B, probe) = {cos(B, h):+.3f}", font_size=24,
                 color=ORANGE),
            Text(f"gap = {cos(A, h) - cos(B, h):.3f}", font_size=26,
                 color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        box.move_to(np.array([4.5, 1.4, 0.0]))
        rect = SurroundingRectangle(box, color=GREY_B, buff=0.18)
        self.play(FadeIn(box), Create(rect))
        self.wait(1.4)

        # --- the projection ---------------------------------------------
        axis = DashedLine(v2p(-3.0 * h), v2p(3.0 * h), color=GREEN,
                          stroke_width=3, dash_length=0.12)
        axis_lbl = Text("orthogonal to disease", font_size=20,
                        color=GREEN).next_to(axis.get_end(), DOWN, buff=0.15)
        self.play(Create(axis), FadeIn(axis_lbl))
        self.wait(0.6)

        drop_a = DashedLine(v2p(A), v2p(Ap), color=YELLOW, stroke_opacity=0.6)
        drop_b = DashedLine(v2p(B), v2p(Bp), color=ORANGE, stroke_opacity=0.6)
        self.play(Create(drop_a), Create(drop_b))
        self.wait(0.4)

        ap_arrow = Arrow(v2p([0, 0]), v2p(Ap), buff=0, color=YELLOW, stroke_width=7)
        bp_arrow = Arrow(v2p([0, 0]), v2p(Bp), buff=0, color=ORANGE, stroke_width=7)
        self.play(
            Transform(a_arrow, ap_arrow),
            Transform(b_arrow, bp_arrow),
            FadeOut(a_lbl), FadeOut(b_lbl),
            FadeOut(drop_a), FadeOut(drop_b),
            run_time=2.0,
        )
        self.wait(0.5)

        # --- scores after -------------------------------------------------
        box2 = VGroup(
            Text(f"cos(A', probe) = {cos(Ap, h):+.3f}", font_size=24,
                 color=YELLOW),
            Text(f"cos(B', probe) = {cos(Bp, h):+.3f}", font_size=24,
                 color=ORANGE),
            Text(f"gap = {cos(Ap, h) - cos(Bp, h):.3f}", font_size=26,
                 color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        box2.move_to(np.array([4.5, 1.4, 0.0]))
        self.play(Transform(box, box2))
        self.wait(1.2)

        punch = Text("same truth — now separable",
                     font_size=30, color=GREEN).to_edge(DOWN)
        self.play(FadeOut(axis_lbl), FadeIn(punch))
        self.wait(2.0)

        # --- closing note --------------------------------------------------
        self.play(FadeOut(punch))
        last = Text("nothing was deleted: 384 numbers before, 384 after —\n"
                    "they just now lie in a 377-dimensional slice",
                    font_size=24, color=WHITE).to_edge(DOWN)
        self.play(FadeIn(last))
        self.wait(2.5)
