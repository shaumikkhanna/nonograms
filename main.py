import pickle
from pprint import pprint
from collections import Counter


with open("output.pkl", "rb") as file:
    data = pickle.load(file)


# pprint(data) 
# Output is saved in an external text file


answers = []
for v in data.values():
    answers.append(v["answer"])

c = Counter(answers)
print(c) # Counter({1: 234686, 2: 117388, 4: 12630, 0: 11657, 5: 7968, 6: 4134, -1: 2162})



