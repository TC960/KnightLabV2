"""3-D Manim visualisation of the disease-direction projection.

Why 3-D reads better than the 2-D version: in 3-D the orthogonal complement of a
single direction is a genuine PLANE, so you watch the papers fall onto it. That
is the actual operation -- in the real space it is a 377-dimensional slice of
384, which nobody can picture, but the 3->2 case is the same move.

Vectors match toy_projection_demo.py exactly:

    d = [1,1,0]/sqrt(2)     disease direction        (removed)
    h = [1,-1,0]/sqrt(2)    probe "hospital"         (lies in the plane)
    l = [0,0,1]             probe "longitudinal"     (lies in the plane)
    A = 3d + 1.0h + 0.4l    up-camp paper
    B = 3d + 0.3h + 1.0l    down-camp paper

DELIBERATELY, A and B do NOT land on a probe after projection. The 2-D version
of this animation was misleading exactly there: in 2-D the orthogonal complement
of one direction is a LINE, and that line coincided with the probe, so
"projecting out disease" and "pointing at the probe" looked like the same arrow.
They are different operations:

    PROJECTION  happens once, globally, and MOVES the vectors into the plane
    SCORING     happens per probe, and only MEASURES an angle -- nothing moves

Here the papers land somewhere in the plane, and the probes are separate rulers
lying in that same plane. Scoring is the angle between them.

    manim -ql manim_projection_3d.py DiseaseProjection3D
    manim -qh manim_projection_3d.py DiseaseProjection3D
"""
import numpy as np
from manim import (
    Arrow3D, BLUE, Create, DEGREES, DashedLine, FadeIn, FadeOut, GREEN, GREY_B,
    LEFT, ORANGE, Surface, Text, ThreeDAxes, ThreeDScene, UP, DOWN, VGroup, WHITE,
    YELLOW, RED, PI,
)

SC = 1.35                    # big enough to read, small enough not to clip


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# GEOMETRY CHOICE. An earlier version put the disease direction at [1,1,0], which
# makes its orthogonal complement a VERTICAL plane containing the z-axis. No
# camera angle renders that intuitively -- face-on hides the out-of-plane offset,
# edge-on hides the plane.
#
# Here disease is mostly-but-not-exactly vertical. Its complement is then a
# near-horizontal plane, so the animation reads as the thing it actually is:
# papers floating above a floor, dropping onto it. Disease is still a diagonal
# (0.32, 0.27, 0.91) -- not an axis -- which is the point being made.
d = unit([0.35, 0.30, 1.0])

# Two orthonormal directions spanning the complement of d, taken from the null
# space rather than hand-picked, so they are guaranteed to lie in the plane.
_ns = np.linalg.svd(d.reshape(1, 3))[2][1:]
h = unit(_ns[0])             # probe: "recruited from hospital"
l = unit(_ns[1])             # probe: "longitudinal cohort"

# Papers sit well off the plane (disease dominates) and differ in the plane.
# Neither lands ON a probe after projection -- projection MOVES, probes MEASURE.
A = 2.6 * d + 1.00 * h + 0.35 * l
B = 2.6 * d + 0.30 * h + 1.00 * l
P = np.eye(3) - np.outer(d, d)
Ap, Bp = P @ A, P @ B


def cos(a, b):
    return float(unit(a) @ unit(b))


def pt(v):
    return np.array(v, float) * SC


class DiseaseProjection3D(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1], z_range=[-2, 2, 1],
            x_length=5, y_length=5, z_length=4,
        )
        # The camera direction is (sin p cos t, sin p sin t, cos p). The disease
        # direction is [1,1,0]/sqrt(2), i.e. azimuth 45 deg, elevation 0. Looking
        # nearly ALONG it (theta=45, phi near 90) presents its orthogonal plane
        # face-on; back off a little so the scene still reads as 3-D.
        # Tradeoff: looking ALONG the disease direction (theta=45) shows the
        # plane face-on but foreshortens the out-of-plane component to nothing,
        # so the papers look like they are already in the plane. Looking across
        # it (theta<20) shows the offset but renders the plane edge-on. ~30 deg
        # keeps both readable.
        self.set_camera_orientation(phi=62 * DEGREES, theta=-55 * DEGREES,
                                    zoom=1.0)
        self.play(Create(axes), run_time=1.5)

        title = Text("Projecting out the disease direction", font_size=32)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title))

        # --- the disease direction ---------------------------------------
        d_arr = Arrow3D(pt([0, 0, 0]), pt(d * 2.0), color=GREY_B,
                        thickness=0.022, base_radius=0.07)
        d_txt = Text("disease direction", font_size=24, color=GREY_B)
        d_txt.to_corner(UP + LEFT).shift(DOWN * 0.9)
        self.add_fixed_in_frame_mobjects(d_txt)
        self.play(Create(d_arr), FadeIn(d_txt))
        self.begin_ambient_camera_rotation(rate=0.035)
        self.wait(1.5)

        # --- the plane orthogonal to it -----------------------------------
        def plane_fn(u, v):
            return (h * u + l * v) * SC

        plane = Surface(
            plane_fn, u_range=[-1.8, 1.8], v_range=[-1.5, 1.5],
            resolution=(10, 10), fill_opacity=0.16,
            checkerboard_colors=[GREEN, GREEN],   # None breaks Surface in 0.21
            stroke_color=GREEN, stroke_width=1,
        )
        p_txt = Text("the plane orthogonal to it\n(everything that is NOT disease)",
                     font_size=22, color=GREEN)
        p_txt.to_corner(UP + LEFT).shift(DOWN * 1.7)
        self.add_fixed_in_frame_mobjects(p_txt)
        self.play(Create(plane), FadeIn(p_txt), run_time=2.0)
        self.wait(1.5)

        # --- the two papers ------------------------------------------------
        a_arr = Arrow3D(pt([0, 0, 0]), pt(A), color=YELLOW, thickness=0.024,
                        base_radius=0.08)
        b_arr = Arrow3D(pt([0, 0, 0]), pt(B), color=ORANGE, thickness=0.024,
                        base_radius=0.08)
        ab_txt = Text("paper A — ENRICHED        paper B — DEPLETED",
                      font_size=24)
        ab_txt[:18].set_color(YELLOW)
        ab_txt[18:].set_color(ORANGE)
        ab_txt.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(ab_txt)
        self.play(Create(a_arr), Create(b_arr), FadeIn(ab_txt))
        self.wait(1.0)

        score = Text(f"angle between them: {np.degrees(np.arccos(cos(A, B))):.0f}°"
                     f"     —  nearly parallel", font_size=26, color=WHITE)
        score.to_edge(DOWN).shift(UP * 0.55)
        self.add_fixed_in_frame_mobjects(score)
        self.play(FadeIn(score))
        self.wait(2.0)

        # --- drop them onto the plane ---------------------------------------
        drop_a = DashedLine(pt(A), pt(Ap), color=YELLOW, stroke_width=3)
        drop_b = DashedLine(pt(B), pt(Bp), color=ORANGE, stroke_width=3)
        self.play(Create(drop_a), Create(drop_b))
        self.wait(0.8)

        ap_arr = Arrow3D(pt([0, 0, 0]), pt(Ap), color=YELLOW, thickness=0.024,
                         base_radius=0.08)
        bp_arr = Arrow3D(pt([0, 0, 0]), pt(Bp), color=ORANGE, thickness=0.024,
                         base_radius=0.08)
        self.play(FadeOut(a_arr), FadeOut(b_arr), FadeOut(drop_a), FadeOut(drop_b),
                  Create(ap_arr), Create(bp_arr), run_time=2.0)

        score2 = Text(f"angle between them: "
                      f"{np.degrees(np.arccos(cos(Ap, Bp))):.0f}°"
                      f"     —  fully separated", font_size=26, color=GREEN)
        score2.to_edge(DOWN).shift(UP * 0.55)
        self.add_fixed_in_frame_mobjects(score2)
        self.play(FadeOut(score), FadeIn(score2))
        self.wait(2.0)

        # --- the probes live in the plane -------------------------------------
        h_arr = Arrow3D(pt([0, 0, 0]), pt(h * 2.4), color=BLUE, thickness=0.018,
                        base_radius=0.06)
        l_arr = Arrow3D(pt([0, 0, 0]), pt(l * 2.0), color=RED, thickness=0.018,
                        base_radius=0.06)
        pr = Text('probes: "recruited from hospital"   and   "longitudinal cohort"',
                  font_size=22)
        pr[:34].set_color(BLUE)
        pr[34:].set_color(RED)
        pr.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(pr)
        self.play(FadeOut(ab_txt), Create(h_arr), Create(l_arr), FadeIn(pr))
        self.wait(1.5)

        final = Text(
            f"projection MOVED them into the plane;  the probes only MEASURE\n"
            f"cos(A', hospital) = {cos(Ap, h):.2f}      "
            f"cos(B', hospital) = {cos(Bp, h):.2f}",
            font_size=24, color=WHITE)
        final.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final)
        self.play(FadeOut(pr), FadeOut(score2), FadeIn(final))
        self.wait(3.0)
        self.stop_ambient_camera_rotation()
