---
icon: "material/wrench"
---

# Installation

`setselector` requires Python 3.12 or newer. Install it from a clone of the
repository with:

```bash
conda create -n <name> python=3.12
conda activate <name>
git clone https://github.com/potassco/benchmark-set-selection-tool.git
cd benchmark-set-selection-tool
pip install .
```

For a development installation, use `pip install -e .`.

## Verify the installation

Run the command-line help and version checks:

```bash
setselector -h
setselector --version
```
