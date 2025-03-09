import random
import numpy as np


FILLED_CHAR = 'F'
EMPTY_CHAR = 'E'
UNKNOWN_CHAR = '?'


def generate_valid_nonogram(size):
    row_sums = [random.randint(0, size) for _ in range(size)]
    row_clues = [[int(x)] for x in row_sums]
    
    col_sums = [0 for _ in range(size)]
    column_indices_not_full = list(range(size))
    for _ in range(sum(row_sums)):
        random_index = random.choice(column_indices_not_full)
        col_sums[random_index] += 1
        if col_sums[random_index] == size:
            column_indices_not_full.remove(random_index)
    
    col_clues = [[int(x)] for x in col_sums]
    
    return row_clues, col_clues


def solve_simply(row_clues, col_clues):
    size = len(row_clues)
    grid = [[UNKNOWN_CHAR for _ in range(size)] for _ in range(size)]
    
    # Fully filled rows/columns
    for i in range(size):
        if sum(row_clues[i]) + len(row_clues[i]) - 1 == size:
            grid[i] = [FILLED_CHAR if c != EMPTY_CHAR else EMPTY_CHAR for c in grid[i]]
        if sum(col_clues[i]) + len(col_clues[i]) - 1 == size:
            for r in range(size):
                grid[r][i] = FILLED_CHAR
    
    # Mark fully known empty rows/columns
    for i in range(size):
        if row_clues[i] == [0]:
            grid[i] = [EMPTY_CHAR] * size
        if col_clues[i] == [0]:
            for r in range(size):
                grid[r][i] = EMPTY_CHAR
    
    # Use overlapping logic
    # for i in range(size):
    #     if row_clues[i] and sum(row_clues[i]) < size:
    #         possible = [UNKNOWN_CHAR] * size
    #         for j in range(row_clues[i][0]):
    #             possible[j] = FILLED_CHAR
    #         for j in range(size - row_clues[i][0], size):
    #             possible[j] = FILLED_CHAR
    #         for j in range(size):
    #             if possible[j] == FILLED_CHAR:
    #                 grid[i][j] = FILLED_CHAR
    
    #     if col_clues[i] and sum(col_clues[i]) < size:
    #         possible = [UNKNOWN_CHAR] * size
    #         for j in range(col_clues[i][0]):
    #             possible[j] = FILLED_CHAR
    #         for j in range(size - col_clues[i][0], size):
    #             possible[j] = FILLED_CHAR
    #         for j in range(size):
    #             if possible[j] == FILLED_CHAR:
    #                 grid[j][i] = FILLED_CHAR
    
    return grid


def print_grid(grid):
    """Displays the current state of the grid."""
    for row in grid:
        print(" ".join(row))
    print()

# Example Usage
size = 15
row_clues, col_clues = generate_valid_nonogram(size)
grid = solve_simply(row_clues, col_clues)

print("Generated Nonogram Clues:")
print("Row Clues:", row_clues)
print("Column Clues:", col_clues)
print("\nSolved Grid (As Much As Possible):")
print_grid(grid)
