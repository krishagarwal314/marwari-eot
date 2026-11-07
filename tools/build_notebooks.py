"""Build notebooks/NN_*.ipynb from notebooks/src/NN_*.py ("# %%" percent format).

Edit the .py sources (clean git diffs), then run:  python tools/build_notebooks.py
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks" / "src", ROOT / "notebooks"


def cells_from(text: str) -> list[dict]:
    cells = []
    for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            src = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": body.splitlines(True)})
    return cells


def main():
    for src in sorted(SRC.glob("[0-9][0-9]_*.py")):
        nb = {"nbformat": 4, "nbformat_minor": 5,
              "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                           "kernelspec": {"name": "python3", "display_name": "Python 3"}},
              "cells": cells_from(src.read_text())}
        dst = OUT / (src.stem + ".ipynb")
        dst.write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
