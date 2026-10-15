"""
Build the model tutorials (notebooks/model_tutorials/NN_*.ipynb): one notebook per model, simplest first.

    cd notebooks/model_tutorials && ../../.venv/bin/python make_tutorials.py [names...]      (rebuilding a notebook drops its outputs)
    ../../.venv/bin/jupyter execute --inplace 00_foundations.ipynb ...     (or run them in Jupyter)

Each notebook follows the same plan:
    1. the idea in words;
    2. the background and the equations, with every symbol explained;
    3. the model built FROM SCRATCH on a tiny example (numpy / PyTorch), step by step;
    4. the same model from our code (src/ml/) on the toy CCD world;
    5. when to use it, and things to try.
They pair with the learning guide docs/overleaf_ml_guide/ml_guide.tex (section numbers given in each notebook).
The content of the notebooks is in tutorials_part1.py and tutorials_part2.py.
"""
from pathlib import Path

import nbformat as nbf

import tutorials_part1
import tutorials_part2

HERE = Path(__file__).resolve().parent


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(src.strip("\n")) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n"))
                   for kind, src in cells]
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = HERE / f"{name}.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]                       # optional: rebuild only these notebooks (e.g. 03_neural_networks)
    for module in (tutorials_part1, tutorials_part2):
        for name, cells in module.NOTEBOOKS.items():
            if not only or name in only:
                print("wrote", build(name, cells))
