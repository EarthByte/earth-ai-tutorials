import sys
import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path = sys.argv[1]
nb = nbf.read(path, as_version=4)
client = NotebookClient(nb, timeout=300, kernel_name="python3")

try:
    client.execute()
    print(f"OK  -- {path}: all cells executed with no errors.")
except CellExecutionError as e:
    print(f"FAIL -- {path}: execution error.")
    print(str(e)[:2000])
    sys.exit(1)

# quick scan for obviously-empty output where a plot/print was expected
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_with_output = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
print(f"     {n_code} code cells, {n_with_output} produced output")

if "--print-text" in sys.argv:
    for c in nb.cells:
        if c.cell_type == "code":
            for out in c.get("outputs", []):
                if out.get("output_type") == "stream":
                    print("     >>", out["text"].strip().replace("\n", "\n        "))
                elif out.get("output_type") == "error":
                    print("     !! ERROR:", out.get("ename"), out.get("evalue"))
