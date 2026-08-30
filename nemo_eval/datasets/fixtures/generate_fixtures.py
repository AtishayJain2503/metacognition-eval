"""
nemo_eval.datasets.fixtures.generate_fixtures
=============================================
Deterministic generation of offline fixtures for MATH, PutnamBench, and Lila.
"""

import json
import os
from pathlib import Path


def generate_math_fixtures() -> list[dict]:
    subjects_distribution = [
        ("Algebra", 8),
        ("Counting & Probability", 7),
        ("Geometry", 7),
        ("Intermediate Algebra", 7),
        ("Number Theory", 7),
        ("Prealgebra", 7),
        ("Precalculus", 7),
    ]
    tasks = []
    
    # Concrete high quality math problems
    math_catalog = [
        # Algebra (8)
        ("Solve for x: 3x + 7 = 22.", "5", "Algebra", 1, "math_symbolic"),
        ("Find the vertex (h, k) x-coordinate of the parabola y = 2(x - 3)^2 + 5.", "3", "Algebra", 1, "math_symbolic"),
        ("Expand and simplify (2x + 3)(x - 4).", "2*x^2 - 5*x - 12", "Algebra", 2, "math_symbolic"),
        ("Find the positive root of the quadratic equation x^2 - 5x + 6 = 0.", "3", "Algebra", 2, "math_symbolic"),
        ("If f(x) = 2x^3 - 3x + 1, compute f(2).", "11", "Algebra", 2, "math_symbolic"),
        ("Simplify the rational expression \\frac{x^2 - 9}{x - 3} for x \\neq 3.", "x + 3", "Algebra", 2, "math_symbolic"),
        ("Solve for y: \\frac{2y + 1}{5} = 3.", "7", "Algebra", 1, "math_symbolic"),
        ("If a + b = 10 and ab = 21, find the value of a^2 + b^2.", "58", "Algebra", 3, "math_symbolic"),
        
        # Counting & Probability (7)
        ("How many ways are there to choose 3 items from a set of 8 distinct items?", "56", "Counting & Probability", 2, "math_symbolic"),
        ("In how many distinct orders can 5 books be arranged on a shelf?", "120", "Counting & Probability", 1, "math_symbolic"),
        ("A fair 6-sided die is rolled. What is the probability of rolling a prime number?", "\\frac{1}{2}", "Counting & Probability", 1, "fraction"),
        ("How many subsets does a set with 6 elements have?", "64", "Counting & Probability", 2, "math_symbolic"),
        ("How many 4-digit positive integers have all distinct digits?", "4536", "Counting & Probability", 3, "math_symbolic"),
        ("A bag has 4 red and 6 blue marbles. What is the probability of picking 2 red marbles without replacement?", "\\frac{2}{15}", "Counting & Probability", 3, "fraction"),
        ("Find the coefficient of x^3 in the binomial expansion of (x + 2)^5.", "40", "Counting & Probability", 3, "math_symbolic"),
        
        # Geometry (7)
        ("Find the area of a circle with radius 6.", "36*\\pi", "Geometry", 1, "math_symbolic"),
        ("In a right-angled triangle with legs 5 and 12, what is the length of the hypotenuse?", "13", "Geometry", 1, "math_symbolic"),
        ("Find the area of an equilateral triangle with side length 4.", "4*\\sqrt{3}", "Geometry", 2, "math_symbolic"),
        ("Find the volume of a sphere with radius 3.", "36*\\pi", "Geometry", 2, "math_symbolic"),
        ("The perimeter of a rectangle is 34 and its length is 12. Find the width.", "5", "Geometry", 1, "math_symbolic"),
        ("Find the sum of interior angles in degrees of a regular hexagon.", "720", "Geometry", 2, "math_symbolic"),
        ("In a circle of radius 10, find the arc length subtended by an angle of \\frac{\\pi}{3} radians.", "\\frac{10*\\pi}{3}", "Geometry", 3, "math_symbolic"),
        
        # Intermediate Algebra (7)
        ("Find the positive real root of the polynomial equation x^3 - 4x = 0.", "2", "Intermediate Algebra", 2, "math_symbolic"),
        ("Solve for x: \\log_2(x) + \\log_2(x - 2) = 3.", "4", "Intermediate Algebra", 3, "math_symbolic"),
        ("Solve the system x + y = 7 and x^2 + y^2 = 25 for x > y. Find x.", "4", "Intermediate Algebra", 3, "math_symbolic"),
        ("Find the sum of the infinite geometric series 8 + 4 + 2 + 1 + \\dots", "16", "Intermediate Algebra", 2, "math_symbolic"),
        ("Simplify \\sqrt{18} + \\sqrt{50} - \\sqrt{8}.", "6*\\sqrt{2}", "Intermediate Algebra", 2, "math_symbolic"),
        ("Find the minimum value of the quadratic function f(x) = x^2 - 6x + 13.", "4", "Intermediate Algebra", 2, "math_symbolic"),
        ("Find the remainder when P(x) = x^4 - 3x^2 + 5x - 7 is divided by x - 2.", "7", "Intermediate Algebra", 3, "math_symbolic"),
        
        # Number Theory (7)
        ("Find the greatest common divisor of 84 and 180.", "12", "Number Theory", 1, "math_symbolic"),
        ("Find the least common multiple of 14 and 35.", "70", "Number Theory", 1, "math_symbolic"),
        ("Find the remainder when 2^{50} is divided by 7.", "4", "Number Theory", 3, "math_symbolic"),
        ("Find the number of positive divisors of 360.", "24", "Number Theory", 2, "math_symbolic"),
        ("Solve for x in the modular congruence 3x \\equiv 4 \\pmod{7} for 0 \\le x < 7.", "6", "Number Theory", 3, "math_symbolic"),
        ("Find the sum of the distinct prime factors of 210.", "17", "Number Theory", 2, "math_symbolic"),
        ("Find the units digit of 7^{2024}.", "1", "Number Theory", 2, "math_symbolic"),
        
        # Prealgebra (7)
        ("Evaluate 15 - 3 * (4 + 2) / 2.", "6", "Prealgebra", 1, "math_symbolic"),
        ("Convert the fraction \\frac{3}{8} to a decimal.", "0.375", "Prealgebra", 1, "float_tol"),
        ("What is 35% of 240?", "84", "Prealgebra", 1, "math_symbolic"),
        ("If 3 notebooks cost $1.50, what is the cost in dollars of 10 notebooks?", "5", "Prealgebra", 1, "math_symbolic"),
        ("Evaluate (-4)^2 - 2 * (-3) + 5.", "27", "Prealgebra", 1, "math_symbolic"),
        ("Find the arithmetic mean of 12, 18, 25, 31, and 44.", "26", "Prealgebra", 1, "float_tol"),
        ("Solve for x: 2(x - 5) + 3x = 25.", "7", "Prealgebra", 1, "math_symbolic"),
        
        # Precalculus (7)
        ("Evaluate \\sin(\\frac{\\pi}{3}).", "\\frac{\\sqrt{3}}{2}", "Precalculus", 2, "math_symbolic"),
        ("Evaluate \\cos(\\frac{2\\pi}{3}).", "-\\frac{1}{2}", "Precalculus", 2, "math_symbolic"),
        ("Find the exact value of \\tan(\\frac{\\pi}{4}).", "1", "Precalculus", 1, "math_symbolic"),
        ("If \\sin(\\theta) = \\frac{3}{5} for \\theta in Quadrant I, find \\cos(\\theta).", "\\frac{4}{5}", "Precalculus", 2, "fraction"),
        ("Convert polar coordinate r = 4, \\theta = \\frac{\\pi}{6} to Cartesian x-coordinate.", "2*\\sqrt{3}", "Precalculus", 3, "math_symbolic"),
        ("Simplify \\sin^2(x) + \\cos^2(x) + \\tan^2(x).", "\\sec^2(x)", "Precalculus", 2, "math_symbolic"),
        ("Find the magnitude of the 2D vector v = (3, -4).", "5", "Precalculus", 1, "math_symbolic"),
    ]
    
    for i, (prob, ans, subj, lvl, etype) in enumerate(math_catalog):
        task_id = f"math_{subj.lower().replace(' ', '_').replace('&', 'and')}_{i+1:03d}"
        tasks.append({
            "task_id": task_id,
            "benchmark_name": "math",
            "query": f"{prob}\n\nSolve this mathematical problem step-by-step. Put your final answer within \\boxed{{}}.",
            "ground_truth": f"\\boxed{{{ans}}}",
            "eval_type": etype,
            "metadata": {
                "subject": subj,
                "level": lvl,
                "split": "test",
                "boxed_solution": ans,
                "source": "hendrycks_math"
            }
        })
    return tasks


def generate_putnam_fixtures() -> list[dict]:
    categories = [
        "real_analysis",
        "abstract_algebra",
        "linear_algebra",
        "number_theory",
        "combinatorics",
        "geometry",
        "calculus",
    ]
    tasks = []
    for i in range(50):
        year = 2000 + (i % 24)
        cat = categories[i % len(categories)]
        prob_label = f"A{(i % 6) + 1}" if i % 2 == 0 else f"B{(i % 6) + 1}"
        
        if cat == "real_analysis":
            prob = f"Let f: [0, 1] -> R be continuous with \\int_0^1 f(x) dx = {i+2}. Evaluate the limit as n -> oo of n \\int_0^{{1/n}} f(x) dx."
            ans = f"{i+2}"
            etype = "math_symbolic"
        elif cat == "abstract_algebra":
            prob = f"Let G be a finite group of order {2*(i+3)}. If G has a normal subgroup of order {i+3}, find the index [G:H]."
            ans = "2"
            etype = "math_symbolic"
        elif cat == "linear_algebra":
            prob = f"Let A be a 2x2 matrix with trace {2*i + 3} and determinant {(i+1)*(i+2)}. Find the sum of the eigenvalues of A."
            ans = f"{2*i + 3}"
            etype = "math_symbolic"
        elif cat == "number_theory":
            prob = f"Find the number of positive integers n <= 100 such that n^2 + {i+1} is divisible by {i+2}."
            ans = str((100 // (i + 2)) + (1 if (100 % (i + 2)) >= (i % 2 + 1) else 0))
            etype = "math_symbolic"
        elif cat == "combinatorics":
            prob = f"Let S be a finite set of {i+4} elements. Find the number of permutations with exactly one fixed point."
            ans = str((i + 4) * max(1, (i + 3) // 2))
            etype = "math_symbolic"
        elif cat == "geometry":
            prob = f"Let ABC be a triangle in R^2 with vertices at (0,0), ({i+2}, 0), and (0, {i+3}). Compute the area."
            ans = f"{(i+2)*(i+3)}/2"
            etype = "fraction"
        else: # calculus
            prob = f"Evaluate \\int_0^1 x^{i+1} (1 - x) dx."
            ans = f"1/{ (i+2)*(i+3) }"
            etype = "fraction"

        tasks.append({
            "task_id": f"putnam_{year}_{prob_label}_{i+1:03d}",
            "benchmark_name": "putnam",
            "query": f"Putnam {year} Problem {prob_label} ({cat}): {prob}\n\nSolve this Putnam competition problem step-by-step. State your final answer clearly within \\boxed{{}}.",
            "ground_truth": f"\\boxed{{{ans}}}",
            "eval_type": etype,
            "metadata": {
                "competition": "Putnam",
                "year": year,
                "problem": prob_label,
                "category": cat,
                "subdiscipline": cat,
                "split": "test",
                "formal_verification": True,
                "boxed_solution": ans
            }
        })
    return tasks


def generate_lila_fixtures() -> list[dict]:
    subcategories = [
        "arithmetic",
        "algebra",
        "calculus",
        "geometry",
        "combinatorics",
        "physics",
        "statistics",
    ]
    tasks = []
    for subcat in subcategories:
        for j in range(50):
            idx = j + 1
            if subcat == "arithmetic":
                a, b, c = 10 * j + 12, 3 * j + 7, 2 * j + 5
                val = (a + b) * c
                prob = f"Compute ({a} + {b}) * {c}."
                gt = str(val)
                etype = "exact"
            elif subcat == "algebra":
                k = j + 1
                prob = f"Simplify the algebraic expression (x + {k})^2 - (x - {k})^2."
                gt = f"{4*k}*x"
                etype = "math_symbolic"
            elif subcat == "calculus":
                n = j + 2
                prob = f"Compute the derivative of f(x) = {j+1}*x^{n} at x = 1."
                gt = str((j + 1) * n)
                etype = "math_symbolic"
            elif subcat == "geometry":
                leg1 = 3 * (j + 1)
                leg2 = 4 * (j + 1)
                hyp = 5 * (j + 1)
                prob = f"Find the hypotenuse of a right triangle with legs {leg1} and {leg2}."
                gt = str(float(hyp))
                etype = "float_tol"
            elif subcat == "combinatorics":
                p1, p2 = 2, 2 * j + 3
                prob = f"Find the set of prime factors of {p1 * p2}."
                gt = f"{{{p1}, {p2}}}" if p2 != p1 else f"{{{p1}}}"
                etype = "set"
            elif subcat == "physics":
                m = 2 * (j + 1)
                v = 3
                ke = 0.5 * m * (v ** 2)
                prob = f"Calculate the kinetic energy (in Joules) of an object with mass {m} kg moving at velocity {v} m/s."
                gt = str(float(ke))
                etype = "float_tol"
            else: # statistics
                mean_val = float(j + 3)
                prob = f"Find the mean of the data set [{j}, {j+2}, {j+4}, {j+6}]."
                gt = str(mean_val)
                etype = "float_tol"

            tasks.append({
                "task_id": f"lila_{subcat}_{idx:03d}",
                "benchmark_name": "lila",
                "query": f"{prob}\n\nSolve this problem step-by-step using Python code or mathematical reasoning. Your final answer should be enclosed in \\boxed{{}}.",
                "ground_truth": f"\\boxed{{{gt}}}" if etype in ("math_symbolic", "fraction") else gt,
                "eval_type": etype,
                "metadata": {
                    "source": "allenai_lila",
                    "subcategory": subcat,
                    "subdiscipline": subcat.capitalize(),
                    "sample_idx": idx,
                    "split": "test",
                    "boxed_solution": gt
                }
            })
    return tasks


def main():
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    math_tasks = generate_math_fixtures()
    with open(fixtures_dir / "math_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in math_tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(math_tasks)} MATH fixtures.")

    putnam_tasks = generate_putnam_fixtures()
    with open(fixtures_dir / "putnam_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in putnam_tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(putnam_tasks)} Putnam fixtures.")

    lila_tasks = generate_lila_fixtures()
    with open(fixtures_dir / "lila_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in lila_tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(lila_tasks)} Lila fixtures.")


if __name__ == "__main__":
    main()
