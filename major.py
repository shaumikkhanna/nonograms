import random
import numpy as np
import sys
from itertools import product
from random import choice
import pickle
from collections import Counter


def generate_valid_clues_old(size):
    """
    Generates a valid nonogram of given size. Returns a tuple of row clues and column clues.
    """
    row_sums = [random.randint(0, size) for _ in range(size)]
    row_clues = [int(x) for x in row_sums]
    
    col_sums = [0 for _ in range(size)]
    column_indices_not_full = list(range(size))
    for _ in range(sum(row_sums)):
        random_index = random.choice(column_indices_not_full)
        col_sums[random_index] += 1
        if col_sums[random_index] == size:
            column_indices_not_full.remove(random_index)
    
    col_clues = [int(x) for x in col_sums]

    # assert sum(row_clues) == sum(col_clues)
    return row_clues, col_clues


def generate_valid_clue(size):
    """
    This function generates clues for a nonogram of given size.
    It ensures that the clues are valid by checking the following conditions:
    1. The sum of row clues must equal the sum of column clues.
    2. The clues must not be the same when transposed.
    3. The clues must not be the same when flipped.
    4. If a row or column is empty, then there cannot be a full column or row.
    """
    while True:
        clues = np.random.randint(0, size + 1, 2*size)
        row_clues, col_clues = tuple(clues[:size]), tuple(clues[size:])

        # Rows and columns must have the same number of black cells
        if sum(row_clues) != sum(col_clues):
            continue
        # This accounts for the fact that a traspose of a nonogram will be duplicated
        if row_clues > col_clues:
            continue
        # This accounts for the fact that a nonogram flipped will be duplicated
        if row_clues > row_clues[::-1] or col_clues > col_clues[::-1]:
            continue
        # If a row / column is empty, then there is no column / row that is full
        if any(x == 0 for x in row_clues) and any(x == size for x in col_clues):
            continue
        # If a row / column is full, then there is no column / row that is empty
        if any(x == size for x in row_clues) and any(x == 0 for x in col_clues):
            continue

        return row_clues, col_clues


def generate_all_valid_clues(size):
    """
    Generates all valid nonograms of given size. 
    Returns a list of tuples of row clues and column clues.
    """
    for clues in product(range(size+1), repeat=2*size):
        row_clues, col_clues = clues[:size], clues[size:]

        # Rows and columns must have the same number of black cells
        if sum(row_clues) != sum(col_clues):
            continue
        # This accounts for the fact that a traspose of a nonogram will be duplicated
        if row_clues > col_clues:
            continue
        # This accounts for the fact that a nonogram flipped will be duplicated
        if row_clues > row_clues[::-1] or col_clues > col_clues[::-1]:
            continue
        # If a row / column is empty, then there is no column / row that is full
        if any(x == 0 for x in row_clues) and any(x == size for x in col_clues):
            continue
        # If a row / column is full, then there is no column / row that is empty
        if any(x == size for x in row_clues) and any(x == 0 for x in col_clues):
            continue

        yield row_clues, col_clues


def grid_string(grid, row_clues, col_clues):
    """
    Converts a nonogram grid to a string representation.
    """
    output = ""

    max_row_clue_len = max(len(str(clue)) for clue in row_clues)
    max_col_clue_len = max(len(str(clue)) for clue in col_clues)

    # Print column clues
    output += " " * (max_row_clue_len + 1)
    for col in col_clues:
        output += f"{col:>{max_col_clue_len}} "
    output += "\n"

    # Print row clues and grid
    for i, row in enumerate(grid):
        output += f"{row_clues[i]:>{max_row_clue_len}} "
        for cell in row:
            if cell == 1:
                output += "■ "
            elif cell == 0:
                output += "□ "
            elif cell == -1:
                output += "X "
            else:
                raise ValueError("Invalid cell value")
        output += "\n"

    return output


def is_nonogram_contradiction(row_clues, col_clues, black_grid, crossed_grid):
    n = black_grid.shape[0]

    # Step 1: Check for overlapping black and crossed cells
    xs, ys = np.where(black_grid + crossed_grid >= 2)
    if len(xs):
        return True, f'Issue in row {xs[0]+1} column {ys[0]+1}' # A cell is both black and crossed out


    # Helper function to check a single line (row or column)
    def check_line(line, crossed, clue):
        cross_count = np.sum(crossed)

        # Step 2: Too many blacks
        if np.sum(line) > clue:
            return True
        # Step 3: Not enough available cells
        if np.sum(crossed) + clue > n:
            return True

        # Step 4: Contiguous block check
        # Find all valid windows of size `clue` with no crosses
        for start in range(n - clue + 1):
            window = slice(start, start + clue)
            if np.any(crossed[window]):
                continue
            # Are all black cells within this window or outside the line?
            if np.all((line == 0) | ((np.arange(n) >= start) & (np.arange(n) < start + clue))):
                return False  # This window is valid
        
        return True  # No valid window found


    # Check rows
    for i in range(n):
        if check_line(black_grid[i], crossed_grid[i], row_clues[i]):
            return True, f'Issue in row {i+1}'

    # Check columns
    for j in range(n):
        if check_line(black_grid[:, j], crossed_grid[:, j], col_clues[j]):
            return True, f'Issue in column {j+1}'

    return False, ''


def solve_nonogram_line(line, clue):
    n = len(line)
    forced_black = np.zeros(n, dtype=bool)
    forced_cross = np.zeros(n, dtype=bool)

    black_indices = np.where(line == 1)[0]
    cross_indices = np.where(line == -1)[0]
    empty_mask = (line == 0)

    # Step 0: If clue is already satisfied, cross out the rest
    if len(black_indices) == clue:
        forced_cross[empty_mask] = True
        return forced_black.astype(int), forced_cross.astype(int)

    # Step 1: Identify empty intervals between crosses
    intervals = []
    start = 0
    while start < n:
        while start < n and line[start] == -1:
            start += 1
        if start >= n:
            break
        end = start
        while end < n and line[end] != -1:
            end += 1
        intervals.append((start, end))
        start = end

    # Step 2: Special case — line completely empty
    if len(black_indices) == 0 and len(cross_indices) == 0:
        for start, end in intervals:
            length = end - start
            if length >= clue:
                overlap = length - clue
                forced_black[start + overlap : end - overlap] = True
        return forced_black.astype(int), forced_cross.astype(int)

    # Step 3: Fill between existing black cells
    if len(black_indices) > 1:
        forced_black[black_indices[0]:black_indices[-1]+1] = True

    # Step 4: Cross out intervals too short
    for start, end in intervals:
        if end - start < clue:
            forced_cross[start:end] = True

    # Step 5: Forced overlap in only valid interval
    valid_intervals = [iv for iv in intervals if (iv[1] - iv[0] >= clue)]
    if len(valid_intervals) == 1:
        start, end = valid_intervals[0]
        length = end - start
        overlap = length - clue
        for i in range(start + overlap, end - overlap):
            if line[i] != -1:
                forced_black[i] = True

    # Step 6: Cross out cells too far from known filled blocks
    if len(black_indices) > 0:
        # Identify contiguous black blocks
        blocks = []
        start = black_indices[0]
        for i in range(1, len(black_indices)):
            if black_indices[i] != black_indices[i-1] + 1:
                blocks.append((start, black_indices[i-1]))
                start = black_indices[i]
        blocks.append((start, black_indices[-1]))

        for start, end in blocks:
            block_len = end - start + 1
            margin = clue - block_len
            # Cross out left side
            for i in range(0, max(0, start - margin)):
                if line[i] == 0:
                    forced_cross[i] = True
            # Cross out right side
            for i in range(min(n, end + margin + 1), n):
                if line[i] == 0:
                    forced_cross[i] = True

    return forced_black.astype(int), forced_cross.astype(int)


def solve_nonogram_stepwise(row_clues, col_clues, black_grid, crossed_grid):
    n = black_grid.shape[0]
    updated = True

    while updated:
        updated = False
        grid_state = black_grid - crossed_grid  # 1 = black, -1 = cross, 0 = undecided

        # Check each row
        for i in range(n):
            line = grid_state[i]
            clue = row_clues[i]
            b_forced, x_forced = solve_nonogram_line(line, clue)

            # Update black_grid and crossed_grid if new cells are inferred
            new_black = (b_forced == 1) & (black_grid[i] == 0)
            new_cross = (x_forced == 1) & (crossed_grid[i] == 0)

            if np.any(new_black) or np.any(new_cross):
                black_grid[i][new_black] = 1
                crossed_grid[i][new_cross] = 1
                updated = True

        # Check each column
        grid_state = black_grid - crossed_grid  # recompute in case grid changed
        for j in range(n):
            line = grid_state[:, j]
            clue = col_clues[j]
            b_forced, x_forced = solve_nonogram_line(line, clue)

            new_black = (b_forced == 1) & (black_grid[:, j] == 0)
            new_cross = (x_forced == 1) & (crossed_grid[:, j] == 0)

            if np.any(new_black) or np.any(new_cross):
                black_grid[:, j][new_black] = 1
                crossed_grid[:, j][new_cross] = 1
                updated = True

        # Run contradiction check
        is_contradiction, msg_contradiction = is_nonogram_contradiction(row_clues, col_clues, black_grid, crossed_grid)
        if is_contradiction:
            return 1, msg_contradiction


    # Check if the nonogram is unresolved
    for i in range(n):
        if np.sum(black_grid[i]) != row_clues[i]:
            return 0, f'Row {i+1} is not satisfied'
        if np.sum(black_grid[:, i]) != col_clues[i]:
            return 0, f'Column {i+1} is not satisfied'
    
    return -1, 'Solved'


def solve_all(size):
    """
    Solves all nonograms of given size and returns a dictionary with the results.
    The keys are the answers (0, 1, -1) and the values are lists of dictionaries containing clues and messages.
    """
    output = dict()
    for row_clues, col_clues in generate_all_valid_clues(size):
        black_grid = np.zeros((size, size), dtype=int)
        crossed_grid = np.zeros((size, size), dtype=int)
        answer, message = solve_nonogram_stepwise(row_clues, col_clues, black_grid, crossed_grid)
        output.setdefault(answer, []).append({
            "clues": (row_clues, col_clues),
            "message": message,
            "final_state": grid_string(black_grid - crossed_grid, row_clues, col_clues),
        })
    return output


def save_to_file(size, filename=None):
    output = solve_all(size)

    if filename is None:
        filename = f'output_{size}x{size}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(output, f)


def solve_some(size, num_nonograms=100, include_unsolvable=True, only_undetermined=False):
    output = []
    while len(output) < num_nonograms:
        row_clues, col_clues = generate_valid_clue(size)
        black_grid = np.zeros((size, size), dtype=int)
        crossed_grid = np.zeros((size, size), dtype=int)
        answer, message = solve_nonogram_stepwise(row_clues, col_clues, black_grid, crossed_grid)
        
        if not include_unsolvable and answer == 1:
            continue
        if only_undetermined and answer != 0:
            continue

        output.append({
            "clues": (row_clues, col_clues),
            "message": message,
            "final_state": grid_string(black_grid - crossed_grid, row_clues, col_clues),
        })
    return output



if __name__ == "__main__":
    try:
        size = int(sys.argv[1])
    except IndexError:
        pass

    # save_to_file(size)

    # all_nonograms = solve_all(size)
    # for answer, nonograms in all_nonograms.items():
    #     print("=" * 20)
    #     print(f"Answer: {answer}")
    #     print(f'Number of nonograms: {len(nonograms)}')
    #     for data in nonograms:
    #         print(f"Clues: {data['clues']}")
    #         print(f"Message: {data['message']}")
    #         print(data['final_state'])
    #         print("-" * 20)

