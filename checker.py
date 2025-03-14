import random
import numpy as np
import sys
from itertools import product
from random import choice


def generate_valid_nonogram(size):
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


def print_grid(grid, row_clues, col_clues):
    size = len(row_clues)
    max_row_clue_len = max(len(str(clue)) for clue in row_clues)
    max_col_clue_len = max(len(str(clue)) for clue in col_clues)

    # Print column clues
    print(" " * (max_row_clue_len + 1), end="")
    for col in col_clues:
        print(f"{col:>{max_col_clue_len}}", end=" ")
    print()

    # Print row clues and grid
    for i, row in enumerate(grid):
        print(f"{row_clues[i]:>{max_row_clue_len}} ", end="")
        for cell in row:
            if cell == 1:
                print("■", end=" ")
            elif cell == 0:
                print("□", end=" ")
            elif cell == -1:
                print("X", end=" ")
            else:
                raise ValueError("Invalid cell value")
        print()
    print()


def check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
    return \
        np.any(black_grid + crossed_grid >= 2) or \
        any(sum(black_grid[row_index]) > row_clues[row_index] for row_index in range(len(row_clues))) or \
        any(sum(black_grid[:, col_index]) > col_clues[col_index] for col_index in range(len(col_clues)))


def solve_simply(row_clues, col_clues, black_grid, crossed_grid):
    size = len(row_clues)

    # Fully filled rows/columns
    for i in range(size):
        if row_clues[i] == size:
            black_grid[i] = np.ones(size, dtype=int)
        if col_clues[i] == size:
            black_grid[:, i] = np.ones(size, dtype=int)


    # Fully crossed rows/columns
    for i in range(size):
        if row_clues[i] == 0:
            if np.any(black_grid[i] == 1):
                return "contradiction at step 1 at row " + str(i+1)
            crossed_grid[i] = np.ones(size, dtype=int)
        if col_clues[i] == 0:
            if np.any(black_grid[:, i] == 1):
                return "contradiction at step 1 at column " + str(i+1)
            crossed_grid[:, i] = np.ones(size, dtype=int)


    # Fill in using overlapping logic
    for i in range(size):
        if row_clues[i] > size // 2:
            left, right = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
            left[:row_clues[i]] = 1
            right[-row_clues[i]:] = 1
            overlap = np.where(left + right == 2)[0]
            black_grid[i, overlap[0]:overlap[-1]+1] = 1

        print(f'Overlapping on row {i}') 
        print_grid(black_grid - crossed_grid, row_clues, col_clues)
    if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
        return "contradiction at step 2"

    for i in range(size):
        if col_clues[i] > size // 2:
            left, right = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
            left[:col_clues[i]] = 1
            right[-col_clues[i]:] = 1
            overlap = np.where(left + right == 2)[0]
            black_grid[overlap[0]:overlap[-1]+1, i] = 1

        print(f'Overlapping on column {i}\n')
        print_grid(black_grid - crossed_grid, row_clues, col_clues)
    if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
        return "contradiction at step 2"


    progress_black, progress_crossed = 0, 0
    while True:

        # Cross out rows/columns that are already filled
        for i in range(size):
            if row_clues[i] == sum(black_grid[i]):
                crossed_grid[i] = np.ones(size, dtype=int) - black_grid[i]
            if col_clues[i] == sum(black_grid[:, i]):
                crossed_grid[:, i] = np.ones(size, dtype=int) - black_grid[:, i]
            if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                return "contradiction at step 3 at row/column " + str(i+1)
        print('Filling rows / columns that are satisfied\n')
        print_grid(black_grid - crossed_grid, row_clues, col_clues)


        # Fill in rows/columns that are already crossed out
        for i in range(size):
            if row_clues[i] == size - sum(crossed_grid[i]):
                black_grid[i] = np.ones(size, dtype=int) - crossed_grid[i]
            if col_clues[i] == size - sum(crossed_grid[:, i]):
                black_grid[:, i] = np.ones(size, dtype=int) - crossed_grid[:, i]

            if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                return "contradiction at step 4 row/column " + str(i+1)
        print('Crossing out rows / columns that are satisfied\n')
        print_grid(black_grid - crossed_grid, row_clues, col_clues)
            
        
        # More overlapping logic
        for i in range(size):
            intervals, possibility = [], None

            crossed_indices = list(np.where(crossed_grid[i] == 1)[0])
            if len(crossed_indices) == 0:
                continue

            if crossed_indices[0] != 0:
                intervals.append((0, crossed_indices[0] - 1))
            if crossed_indices[-1] != size - 1:
                intervals.append((crossed_indices[-1] + 1, size - 1))
            if len(crossed_indices) > 1:
                for ind_ in range(len(crossed_indices) - 1):
                    intervals.append((crossed_indices[ind_] + 1, crossed_indices[ind_ + 1] - 1))
            
            for interval in intervals:
                interval_length = interval[1] - interval[0] + 1
                if row_clues[i] <= interval_length:
                    if possibility is None:
                        possibility = interval
                    else:
                        break
            else:
                if possibility is None:
                    continue

                left, right = np.zeros(len(possibility), dtype=int), np.zeros(len(possibility), dtype=int)
                left[:row_clues[i]] = 1
                right[-row_clues[i]:] = 1
                overlap = np.where(left + right == 2)[0] + possibility[0]

                if len(overlap):
                    # print('hello')
                    # print(left, right, overlap, possibility, row_clues[i], black_grid[i], crossed_grid[i])
                    black_grid[i, overlap[0]:overlap[-1]+1] = 1

                if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                    return "contradiction at step 6"

        print_grid(black_grid - crossed_grid, row_clues, col_clues)

        for i in range(size):
            intervals, possibility = [], None

            crossed_indices = list(np.where(crossed_grid[:, i] == 1)[0])
            if len(crossed_indices) == 0:
                continue

            if crossed_indices[0] != 0:
                intervals.append((0, crossed_indices[0] - 1))
            if crossed_indices[-1] != size - 1:
                intervals.append((crossed_indices[-1] + 1, size - 1))
            if len(crossed_indices) > 1:
                for ind_ in range(len(crossed_indices) - 1):
                    intervals.append((crossed_indices[ind_] + 1, crossed_indices[ind_ + 1] - 1))
            
            for interval in intervals:
                interval_length = interval[1] - interval[0] + 1
                if col_clues[i] <= interval_length:
                    if possibility is None:
                        possibility = interval
                    else:
                        break
            else:
                if possibility is None:
                    continue

                left, right = np.zeros(len(possibility), dtype=int), np.zeros(len(possibility), dtype=int)
                left[:col_clues[i]] = 1
                right[-col_clues[i]:] = 1
                overlap = np.where(left + right == 2)[0] + possibility[0]

                if len(overlap):
                    black_grid[overlap[0]:overlap[-1]+1, i] = 1
                print_grid(black_grid - crossed_grid, row_clues, col_clues)

                if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                    return "contradiction at step 6"
            
        
        progress_black_new = np.sum(black_grid)
        progress_crossed_new = np.sum(crossed_grid)
        if progress_black_new == progress_black and progress_crossed_new == progress_crossed:
            break
        progress_black, progress_crossed = progress_black_new, progress_crossed_new

    
    # Check nonogram final
    for i in range(size):
        if row_clues[i] != sum(black_grid[i]):
            return "row " + str(i+1) + " not filled"
        if 1 in black_grid[i]:
            ones_indices = np.where(black_grid[i] == 1)[0]
            if not np.all(np.diff(ones_indices) == 1):
                return "row " + str(i+1) + " has non-contiguous ones"
        if col_clues[i] != sum(black_grid[:, i]):
            return "column " + str(i+1) + " not filled"
        if 1 in black_grid[:, i]:
            ones_indices = np.where(black_grid[:, i] == 1)[0]
            if not np.all(np.diff(ones_indices) == 1):
                return "column " + str(i+1) + " has non-contiguous ones"


def main():
    # size = int(sys.argv[1])
    # for _ in range(10):
    #     row_clues, col_clues = generate_valid_nonogram(size)
    #     black_grid = np.zeros((size, size), dtype=int)
    #     crossed_grid = np.zeros((size, size), dtype=int)

    #     print(f'row_clues: {row_clues}; col_clues: {col_clues}')
    #     print(solve_simply(row_clues, col_clues, black_grid, crossed_grid))
    #     print_grid(black_grid - crossed_grid, row_clues, col_clues)
    #     print('\n\n')

    # for clues in product(range(5), repeat=8):
    # row_clues, col_clues = clues[:4], clues[4:]
    for _ in range(10):
        row_clues, col_clues = generate_valid_nonogram(4)
        black_grid = np.zeros((4, 4), dtype=int)
        crossed_grid = np.zeros((4, 4), dtype=int)

        print(f'row_clues: {row_clues} col_clues: {col_clues}')
        print(solve_simply(row_clues, col_clues, black_grid, crossed_grid))
        print_grid(black_grid - crossed_grid, row_clues, col_clues)
        print('\n\n')


if __name__ == "__main__":
    main()

