#!/usr/bin/env python3
"""A 3-dimensional worked example of the projection + subtraction logic.

Everything the real pipeline does in 384 dimensions, small enough to read.

ONE HONEST SIMPLIFICATION: here the disease direction is [1,1,0]/sqrt(2), which
touches 2 of 3 axes. In the real space it touches essentially all 384 -- its
largest single-axis weight is 0.150, and it takes 167 axes to account for 90% of
it. So the real case is MORE diagonal than this toy, not less. No axis means
"disease" there, and none does here either.

    python toy_projection_demo.py
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


# ---------------------------------------------------------------- the space
# Two "content" directions we care about, and one nuisance direction we don't.
d = unit([1, 1, 0])        # DISEASE direction (diagonal -- not an axis)
h = unit([1, -1, 0])       # "recruited from hospital"   (orthogonal to d)
l = unit([0, 0, 1])        # "longitudinal cohort"       (orthogonal to both)

# Two papers on the SAME contested edge: same taxon, same disease, opposite
# reported direction. Both are dominated by disease content (weight 3), and
# differ in one study-design feature (weight 1).
A = 3 * d + 1 * h          # up-camp paper, discusses hospital recruitment
B = 3 * d + 1 * l          # down-camp paper, discusses longitudinal design

print("=" * 74)
print("STEP 0 -- the vectors")
print("=" * 74)
for n, v in [("disease direction d", d), ("probe: hospital  h", h),
             ("probe: longitudinal l", l), ("paper A (up camp)", A),
             ("paper B (down camp)", B)]:
    print(f"  {n:<22} {np.array2string(v, precision=3)}")

print()
print("=" * 74)
print("STEP 1 -- score each paper against each probe, RAW")
print("=" * 74)
print(f"  {'':<22}{'cos(paper, h)':>16}{'cos(paper, l)':>16}")
print(f"  {'paper A (up)':<22}{cos(A,h):>16.3f}{cos(A,l):>16.3f}")
print(f"  {'paper B (down)':<22}{cos(B,h):>16.3f}{cos(B,l):>16.3f}")
print(f"  {'GAP (up - down)':<22}{cos(A,h)-cos(B,h):>16.3f}"
      f"{cos(A,l)-cos(B,l):>16.3f}")
print("\n  The gaps have the right SIGN but are shrunk: the weight-3 disease")
print("  component sits in both papers and dilutes every cosine toward zero.")

print()
print("=" * 74)
print("STEP 2 -- subtract the two papers (what YOU described)")
print("=" * 74)
diff = A - B
print(f"  A - B = {np.array2string(diff, precision=3)}")
print(f"  h - l = {np.array2string(h - l, precision=3)}")
print(f"  disease component remaining in A-B: {abs(diff @ d):.6f}")
print("\n  Disease cancels EXACTLY, because it was common to both. This is the")
print("  'left with everything but the common factor' intuition, and it holds.")

print()
print("=" * 74)
print("STEP 3 -- project the disease direction out of everything")
print("=" * 74)
P = np.eye(3) - np.outer(d, d)          # projector onto the complement of d
Ap, Bp = P @ A, P @ B
print(f"  projector P = I - d d^T")
print(f"  A' = P A = {np.array2string(Ap, precision=3)}   (was {np.array2string(A, precision=3)})")
print(f"  B' = P B = {np.array2string(Bp, precision=3)}   (was {np.array2string(B, precision=3)})")
print(f"  disease left in A': {abs(Ap @ d):.2e}   in B': {abs(Bp @ d):.2e}")
print("\n  Note NOTHING was deleted -- A' is still 3 numbers. It now lies in the")
print("  2-D plane orthogonal to d. Same as 384 -> a 377-dim slice of 384.")

print()
print("=" * 74)
print("STEP 4 -- rescore against the probes, AFTER projection")
print("=" * 74)
print(f"  {'':<22}{'cos(paper, h)':>16}{'cos(paper, l)':>16}")
print(f"  {'paper A (up)':<22}{cos(Ap,h):>16.3f}{cos(Ap,l):>16.3f}")
print(f"  {'paper B (down)':<22}{cos(Bp,h):>16.3f}{cos(Bp,l):>16.3f}")
print(f"  {'GAP (up - down)':<22}{cos(Ap,h)-cos(Bp,h):>16.3f}"
      f"{cos(Ap,l)-cos(Bp,l):>16.3f}")
print("\n  Gap 0.316 -> 1.000. Same truth, far louder. THIS is why projection is")
print("  not redundant with subtraction: our statistic compares SCORES, and a")
print("  score is an absolute cosine that the shared disease bulk was diluting.")

print()
print("=" * 74)
print("STEP 5 -- why it matters MORE once you aggregate across edges")
print("=" * 74)
d2 = unit([1, 0.2, 0])                 # a second, different disease
C = 3 * d2 + 1 * h                      # paper on ANOTHER edge, also 'hospital'
print(f"  second disease d2 = {np.array2string(d2, precision=3)}")
print(f"  cos(C, h) raw        = {cos(C,h):>6.3f}")
print(f"  cos(A, h) raw        = {cos(A,h):>6.3f}   <- same content, different score")
print(f"  ...because the two diseases sit at different angles to the probe.")
Cp = P @ C
print(f"  cos(C, h) projected  = {cos(Cp,h):>6.3f}")
print(f"  cos(A, h) projected  = {cos(Ap,h):>6.3f}   <- now comparable")
print("\n  Inside ONE edge disease is shared, so it mostly cancels. Pool 43 edges")
print("  spanning different diseases and it stops cancelling -- each disease")
print("  biases its probe scores differently. That is the leak projection plugs.")

print()
print("=" * 74)
print("HOW WE KNOW WHICH DIRECTION TO REMOVE")
print("=" * 74)
print("  We never search for it. We build it from labels we already have:")
print("    real: mean(all Parkinson's chunks), mean(all Alzheimer's chunks), ...")
print("          -> 10 centroids -> centre -> SVD -> keep 95% -> rank-7 basis B")
print("    toy : d is given, and would be recovered as mean(disease papers)")
print("          minus the global mean.")
print("  Then P = I - B B^T, exactly as above but with 7 directions, not 1.")
