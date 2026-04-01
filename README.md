# nn-rns

`nn-rns` packages trained neural networks for evaluating rotating neutron star
observables from an equation-of-state table.

## Install

Install locally while developing:

```bash
pip install -e .
```

Build a wheel and source distribution:

```bash
python -m pip install build
python -m build
```

After this repository is published, users can install it from PyPI with:

```bash
pip install nn-rns
```

Or directly from GitHub with:

```bash
pip install git+https://github.com/<owner>/<repo>.git
```

If you need to build in an offline environment, use:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

## Quick start

```python
from nn_rns import EoSTable, RNSNetworks

eos = EoSTable("path/to/eos_table.rns")
model = RNSNetworks()
model.rns_eval(eos)

print(model.nn_rns_static.shape)
print(model.nn_rns_kepler.shape)
print(model.nn_rns_rotate.shape)
```

The package ships with the trained model weights under `nn_rns/NN/`, so users do
not need to download them separately.

## Development notes

- Source code uses the `src/` layout for reliable packaging.
- Non-Python model assets are included in both wheels and source distributions.
- The legacy `rns_networks` class name is still supported for compatibility.
