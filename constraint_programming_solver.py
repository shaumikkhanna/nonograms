import re
import constraint as cp
import numpy as np
import json
from typing import List, Tuple
from functools import partial


def detect_groups(group_sums, *args):
    str_ = "".join([str(a) for a in list(args)])
    groups = re.split('0+', str_)
    groups = [g for g in groups if g != '']
    result_sums = [len(g) for g in groups]
    if group_sums != result_sums:
        return False
    else:
        return True


def backend(r: int, c: int,
            r_num: List[List[int]],
            c_num: List[List[int]],
            crossed_cells: List[Tuple[int, int]] = None):
    # Setup CP problem object
    problem = cp.Problem()
    
    # Create variables
    variables = [f"A_{row}_{col}" for row in range(r) for col in range(c)]
    problem.addVariables(variables, [0, 1])
    
    # Create starting constraints (crossed-out cells)
    if crossed_cells and len(crossed_cells) != 0:
        crossed_variables = [f"A_{row}_{col}" for row, col in crossed_cells]
        problem.addConstraint(cp.InSetConstraint({0}), crossed_variables)
    
    # Create row-sum and column-sum constraints
    for row in range(r):
        constraint_vars = [v for v in variables if re.search(f"_{row}_", v)]
        row_sum = sum(r_num[row])
        
        # Add constraint to CP object
        # Doesn't work because of https://github.com/python-constraint/python-constraint/issues/48
        # Understand what's happening ToDo
        # problem.addConstraint(lambda *args: sum(args) == row_sum, constraint_vars)
        
        problem.addConstraint(cp.ExactSumConstraint(row_sum), constraint_vars)
        
        # multi-group constraints
        group_details = r_num[row]
        constraint_group = partial(detect_groups, group_details)
        problem.addConstraint(constraint_group, constraint_vars)
        
        # Constraints to speed-up processing
        if row_sum == c:
            # The whole row must be 1's
            problem.addConstraint(cp.InSetConstraint({1}), constraint_vars)
        
    for col in range(c):
        constraint_vars = [v for v in variables if re.search(f"_{col}$", v)]
        col_sum = sum(c_num[col])
        
        # Add constraint to CP object
        # problem.addConstraint(lambda *args: sum(args) == col_sum, constraint_vars)
        problem.addConstraint(cp.ExactSumConstraint(col_sum), constraint_vars)

        # multi-group constraints
        group_details = c_num[col]
        constraint_group = partial(detect_groups, group_details)
        problem.addConstraint(constraint_group, constraint_vars)
        
        # Constraints to speed-up processing
        if col_sum == r:
            # The whole column must be 1's
            problem.addConstraint(cp.InSetConstraint({1}), constraint_vars)
    
    # Solve
    # Convert dict solution into array ToDO
    solutions = problem.getSolutions()
    solutions_array = []
    for solution in solutions:
        variables_in_order = [solution[k] for k in sorted(solution)]
        solutions_array.append(np.array(variables_in_order).reshape((r, c)))
    
    return solutions_array


def normalize_clues(clue_list):
    """Convert list like (1, 2, 0) into [[1], [2], []]"""
    return [[n] if n > 0 else [] for n in clue_list]

def solve_nonogram(puzzle, crossed_cells=None):
    """
    Accepts puzzle format as:
    ((r1, r2, r3...), (c1, c2, c3...))
    where each ri and ci is a single integer (clue sum for that row/column)
    """
    row_ints, col_ints = puzzle
    r_num = normalize_clues(row_ints)
    c_num = normalize_clues(col_ints)
    r, c = len(r_num), len(c_num)
    solutions = backend(r, c, r_num, c_num, crossed_cells)
    if len(solutions) == 0:
        return "No solution", None
    elif len(solutions) == 1:
        return "Unique solution", solutions[0].tolist()
    else:
        return "Multiple solutions", solutions[0].tolist()