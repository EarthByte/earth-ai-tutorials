"""Tiny helper for assembling Jupyter notebooks programmatically.

Usage pattern in each build_sessionNN.py script:

    import nbformat as nbf
    from nb_helper import md, code, save

    cells = [
        md("# Title"),
        code("import numpy as np"),
        ...
    ]
    save(cells, "../session02_bestfit_and_sorting/S02_BestFit_and_Sorting_Machine.ipynb")
"""
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def save(cells, path, kernel_display_name="Python 3"):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": kernel_display_name,
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)
