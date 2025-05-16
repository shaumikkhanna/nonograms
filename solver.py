from constraint_programming_solver import solve_nonogram
import pickle


with open("pickles/output_5x5.pkl", "rb") as f:
    output = pickle.load(f)


for puzzle in output[0]:
    puzzle['label'] = solve_nonogram(puzzle['clues'])[0]


# with open("pickles/output_5x5.pkl", "wb") as f:
#     pickle.dump(output, f)


# for i, nonogram in enumerate(output[0]):
#     status, solution = solve_nonogram(nonogram['clues'])
#     print(f"Puzzle {i+1}: {status}")
#     print(nonogram['final_state'])
#     print()