import random
import numpy as np
import sys


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
    print(black_grid - crossed_grid)

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
    print(black_grid - crossed_grid)


    # Cross out rows/columns that are already filled
    for i in range(size):
        if row_clues[i] == sum(black_grid[i]):
            crossed_grid[i] = np.ones(size, dtype=int) - black_grid[i]
        if col_clues[i] == sum(black_grid[:, i]):
            crossed_grid[:, i] = np.ones(size, dtype=int) - black_grid[:, i]
        if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
            return "contradiction at step 2 at row/column " + str(i+1)
    print(black_grid - crossed_grid)
        

    # Fill in rows/columns that are already crossed out
    for i in range(size):
        if row_clues[i] == size - sum(crossed_grid[i]):
            black_grid[i] = np.ones(size, dtype=int) - crossed_grid[i]
        if col_clues[i] == size - sum(crossed_grid[:, i]):
            black_grid[:, i] = np.ones(size, dtype=int) - crossed_grid[:, i]

        if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
            return "contradiction at step 3 row/column " + str(i+1)
    print(black_grid - crossed_grid)
        

    # Fill in using overlapping logic
    for i in range(size):
        crossed_indices = np.where(crossed_grid[i] == 1)[0]
        possibilites = []
        
        for ind_ in range(len(crossed_indices) - 1):
            interval_length = crossed_indices[ind_ + 1] - crossed_indices[ind_] - 1
            if row_clues[i] <= interval_length < 2*row_clues[i]:
                possibilites.append(ind_)

        if len(possibilites) == 1:
            ind_ = possibilites[0]
            start = crossed_indices[ind_] + 1
            end = crossed_indices[ind_ + 1]
            middle_start = start + (end - start - row_clues[i]) // 2
            middle_end = middle_start + row_clues[i]
            black_grid[i, middle_start:middle_end] = 1

        if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
            return "contradiction at step 4"
        
    for i in range(size):
        crossed_indices = np.where(crossed_grid[:, i] == 1)[0]
        possibilites = []

        for ind_ in range(len(crossed_indices) - 1):
            interval_length = crossed_indices[ind_ + 1] - crossed_indices[ind_] - 1
            if col_clues[i] <= interval_length < 2*col_clues[i]:
                possibilites.append(ind_)

        if len(possibilites) == 1:
            ind_ = possibilites[0]
            start = crossed_indices[ind_] + 1
            end = crossed_indices[ind_ + 1]
            middle_start = start + (end - start - col_clues[i]) // 2
            middle_end = middle_start + col_clues[i]
            black_grid[middle_start:middle_end, i] = 1
        
        if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
            return "contradiction at step 4"
    
    print(black_grid - crossed_grid)
    

    # Cross out rows/columns that are already filled
    for i in range(size):
        if row_clues[i] == sum(black_grid[i]):
            crossed_grid[i] = np.ones(size, dtype=int) - black_grid[i]
        if col_clues[i] == sum(black_grid[:, i]):
            crossed_grid[:, i] = np.ones(size, dtype=int) - black_grid[:, i]

        if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
            return "contradiction at step 5 at row/column " + str(i+1)
        

    # Fill in rows/columns that are already crossed out
    for i in range(size):
        if row_clues[i] == size - sum(crossed_grid[i]):
            black_grid[i] = np.ones(size, dtype=int) - crossed_grid[i]
        if col_clues[i] == size - sum(crossed_grid[:, i]):
            black_grid[:, i] = np.ones(size, dtype=int) - crossed_grid[:, i]

        if check_contradiction(row_clues, col_clues, black_grid, crossed_grid):
            return "contradiction at step 6 row/column " + str(i+1)
        


def main():
    size = int(sys.argv[1])
    # row_clues, col_clues = generate_valid_nonogram(size)
    row_clues, col_clues = [2, 3, 1], [3, 1, 2]
    black_grid = np.zeros((size, size), dtype=int)
    crossed_grid = np.zeros((size, size), dtype=int)

    print(f'row_clues: {row_clues}')
    print(f'col_clues: {col_clues}')
    print(solve_simply(row_clues, col_clues, black_grid, crossed_grid))
    print(black_grid - crossed_grid)

if __name__ == "__main__":
    main()

