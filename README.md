1. Use CONDA to build a python environment:
```bash
conda create -n hyperjet python=3.9
conda activate hyperjet
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Run the code:
```bash
python main.py
```
Note: In order to protect the privacy of the dataset, we only provide the data of ten DAGs, some of the attributes are randomly selected here, and the output main.py the results of the heuristic algorithm.
4. If you want to see the build of the hypergraph, you can run the `generate_hg.py` directly:
```bash
python generate_hg.py
```