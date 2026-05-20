try:
    import nbformat
    use_nbformat = True
except ImportError:
    import json
    use_nbformat = False

file_path = "Benchmark_Invariants.ipynb"
if use_nbformat:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
else:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        
        src = src.replace("import os, time, random, warnings", "import os, time, random, warnings, gc")
        
        old_loop = "                _ = inv_func(A.copy(), B.copy())\n                t1 = time.perf_counter()\n                elapsed = t1 - t0\n                times.append(elapsed)\n                print(f'  {inv_name} iter {it+1}/{ITERATIONS}: {elapsed:.4f}s')"
        new_loop = "                res = inv_func(A.copy(), B.copy())\n                t1 = time.perf_counter()\n                elapsed = t1 - t0\n                times.append(elapsed)\n                print(f'  {inv_name} iter {it+1}/{ITERATIONS}: {elapsed:.4f}s')\n                del res\n                gc.collect()"
        src = src.replace(old_loop, new_loop)
        
        old_end = "            except Exception as e:\n                print(f'  {inv_name} iter {it+1}/{ITERATIONS}: ERROR - {e}')\n                success = False\n                break"
        new_end = old_end + "\n\n        gc.collect()"
        src = src.replace(old_end, new_end)

        if use_nbformat:
            cell["source"] = src
        else:
            new_source = []
            lines = src.split("\n")
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    new_source.append(line + "\n")
                else:
                    if line:
                        new_source.append(line)
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    if use_nbformat:
        nbformat.write(nb, f)
    else:
        json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
