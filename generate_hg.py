from environment.hypergraph import HypergraphData
from utils.utils import get_file_paths

training_paths, test_paths = get_file_paths()
train_hgs = HypergraphData(training_paths)
test_hgs = HypergraphData(test_paths)

for i, hg in enumerate(train_hgs):
    print(f"Hypergraph {i}:")
    for task in hg.tasks:
        print(task)
    for edge in hg.edges:
        print(edge)



