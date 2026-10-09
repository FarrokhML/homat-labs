"""Execute analysis.ipynb top to bottom in place. Fails loudly if any cell errors."""
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("analysis.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "analysis.ipynb")
print("executed without errors:", sum(c.cell_type == "code" for c in nb.cells), "code cells")
