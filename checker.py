import random
import numpy as np
import sys
from itertools import product
from random import choice
import pickle


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


def grid_string(grid, row_clues, col_clues):
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


def check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
    if any(sum(black_grid[row_index]) > row_clues[row_index] for row_index in range(len(row_clues))):
        return True # A row has more black cells than its clue
    
    if any(sum(black_grid[:, col_index]) > col_clues[col_index] for col_index in range(len(col_clues))):
        return True # A column has more black cells than its clue
    
    if np.any(black_grid + crossed_grid >= 2):
        return True # A cell is both black and crossed out
    
    return False


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
                return 1, f"Contradiction at step 1 (row {i+1}) - Intersecting filled column and empty row"
            crossed_grid[i] = np.ones(size, dtype=int)
        if col_clues[i] == 0:
            if np.any(black_grid[:, i] == 1):
                return 1, f"Contradiction at step 1 (column {i+1}) - Intersecting filled row and empty column"
            crossed_grid[:, i] = np.ones(size, dtype=int)


    # Fill in using overlapping logic
    for i in range(size):
        if row_clues[i] > size // 2:
            left, right = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
            left[:row_clues[i]] = 1
            right[-row_clues[i]:] = 1
            overlap = np.where(left + right == 2)[0]
            black_grid[i, overlap[0]:overlap[-1]+1] = 1
    if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
        return 2, "Contradiction at step 2"
    
    for i in range(size):
        if col_clues[i] > size // 2:
            left, right = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
            left[:col_clues[i]] = 1
            right[-col_clues[i]:] = 1
            overlap = np.where(left + right == 2)[0]
            black_grid[overlap[0]:overlap[-1]+1, i] = 1
    if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
        return 2, "Contradiction at step 2"
    
    # print(grid_string(black_grid - crossed_grid, row_clues, col_clues))


    # Loop until no more progress can be made
    progress_black, progress_crossed = np.sum(black_grid), np.sum(crossed_grid)
    loop_number = 0
    while True:

        # STEP 3 -- Cross out rows/columns that are already filled
        for i in range(size):
            if row_clues[i] == sum(black_grid[i]):
                crossed_grid[i] = np.ones(size, dtype=int) - black_grid[i]
            if col_clues[i] == sum(black_grid[:, i]):
                crossed_grid[:, i] = np.ones(size, dtype=int) - black_grid[:, i]
            if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                return 3, f"Contradiction at step 3_{loop_number} at row/column {i+1}"
            

        # STEP 4 -- Fill in rows/columns that are already crossed out
        for i in range(size):
            if row_clues[i] == size - sum(crossed_grid[i]):
                black_grid[i] = np.ones(size, dtype=int) - crossed_grid[i]
            if col_clues[i] == size - sum(crossed_grid[:, i]):
                black_grid[:, i] = np.ones(size, dtype=int) - crossed_grid[:, i]

            if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                return 4, f"Contradiction at step 4_{loop_number} at row/column {i+1}"
                    
        # print("Before overlapping logic:")
        # print(grid_string(black_grid - crossed_grid, row_clues, col_clues))


        # STEP 5 -- More overlapping logic

        # print("Overlapping logic on rows first")
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

                possibility_iterval_length = possibility[1] - possibility[0] + 1
                left, right = np.zeros(possibility_iterval_length, dtype=int), np.zeros(possibility_iterval_length, dtype=int)
                left[:row_clues[i]] = 1
                right[-row_clues[i]:] = 1
                overlap = np.where(left + right == 2)[0] + possibility[0]
                # print(f'hello, left: {left}, right: {right}, overlap: {overlap}, possibility: {possibility}')

                if len(overlap):
                    black_grid[i, overlap[0]:overlap[-1]+1] = 1

                if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                    return 5, f"Contradiction at step 5_{loop_number} at row {i+1}"

            # print("row", i+1); print(grid_string(black_grid - crossed_grid, row_clues, col_clues))

        # print("Overlapping logic on columns next")
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

                possibility_iterval_length = possibility[1] - possibility[0] + 1
                left, right = np.zeros(possibility_iterval_length, dtype=int), np.zeros(possibility_iterval_length, dtype=int)
                left[:col_clues[i]] = 1
                right[-col_clues[i]:] = 1
                overlap = np.where(left + right == 2)[0] + possibility[0]
                # print(f'hello, left: {left}, right: {right}, overlap: {overlap}, possibility: {possibility}')

                if len(overlap):
                    black_grid[overlap[0]:overlap[-1]+1, i] = 1

            if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
                    return 5, f"Contradiction at step 5_{loop_number} at row {i+1}"
            
            # print("column", i+1); print(grid_string(black_grid - crossed_grid, row_clues, col_clues))

        
        progress_black_new = np.sum(black_grid)
        progress_crossed_new = np.sum(crossed_grid)
        if progress_black_new == progress_black and progress_crossed_new == progress_crossed:
            break
        progress_black, progress_crossed = progress_black_new, progress_crossed_new

    
    # Check nonogram final
    for i in range(size):
        if row_clues[i] != sum(black_grid[i]):
            return 0, f"Unsolved - row {i+1} not filled"
        if 1 in black_grid[i]:
            ones_indices = np.where(black_grid[i] == 1)[0]
            if not np.all(np.diff(ones_indices) == 1):
                return 6, f"Contradiction at step 6 - row {i+1} has non-contiguous ones"
        if col_clues[i] != sum(black_grid[:, i]):
            return 0, f"Unsolved - column {i+1} not filled"
        if 1 in black_grid[:, i]:
            ones_indices = np.where(black_grid[:, i] == 1)[0]
            if not np.all(np.diff(ones_indices) == 1):
                return 6, f"Contradiction at step 6 - column {i+1} has non-contiguous ones"
    else:
        return -1, "Solved"


def main():
    # size = int(sys.argv[1])
    size = 4

    output = dict()
    progress_number, total_permutations = 0, (size+1)**(2*size)
    for clues in product(range(size+1), repeat=2*size):
        row_clues, col_clues = clues[:size], clues[size:]
        black_grid = np.zeros((size, size), dtype=int)
        crossed_grid = np.zeros((size, size), dtype=int)
        answer, message = solve_simply(row_clues, col_clues, black_grid, crossed_grid)
        output[clues] = {
            "answer": answer,
            "message": message,
            "final_state": grid_string(black_grid - crossed_grid, row_clues, col_clues),
        }

        if progress_number % 10000 == 0:
            print(f"Progress: {round(progress_number / total_permutations * 100, 2)} %")
        progress_number += 1

    with open('output.pkl', 'wb') as f:
        pickle.dump(output, f)



    # for _ in range(10):
    #     row_clues, col_clues = generate_valid_nonogram(size)
    #     black_grid = np.zeros((size, size), dtype=int)
    #     crossed_grid = np.zeros((size, size), dtype=int)

    #     print(f'row_clues, col_clues = {row_clues}, {col_clues}')
    #     print(solve_simply(row_clues, col_clues, black_grid, crossed_grid))
    #     print(grid_string(black_grid - crossed_grid, row_clues, col_clues))
    #     print()



    # row_clues, col_clues = [3, 1, 0, 4], [2, 2, 2, 2]
    # black_grid = np.zeros((4, 4), dtype=int)
    # crossed_grid = np.zeros((4, 4), dtype=int)

    # print(f'row_clues, col_clues = {row_clues}, {col_clues}')
    # print(solve_simply(row_clues, col_clues, black_grid, crossed_grid))
    # print(grid_string(black_grid - crossed_grid, row_clues, col_clues))
    # print('\n\n')


if __name__ == "__main__":
    main()

