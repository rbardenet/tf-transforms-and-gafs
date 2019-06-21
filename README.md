## hyperbolic

_hyperbolic_ is a Python 3 library implementing time-frequency transforms that map specific white noises to canonical Gaussian analytic functions. _hyperbolic_ is the companion package to the paper
> Rémi Bardenet and Adrien Hardy, Time-frequency transforms of white noises and Gaussian analytic functions, to appear in Applied and Computational Harmonic Analysis (ACHA), [arxiv preprint](https://arxiv.org/abs/1807.11554).

### Install from sources

Clone (or download) this repository

```bash
git clone https://github.com/rbardenet/hyperbolic.git
cd hyperbolic
```

and execute `setup.py`

```bash
pip install --process-dependency-links .
```

If you're in development mode and you want to install also dev packages, documentation and/or tests, you can do as follows:

```bash
pip install -e --process-dependency-links .
```

## Usage examples

You can import *hyperbolic* by doing

```python
import hyperbolic as hype
```

The experiments in the paper are run using the class `Experiment`. Here is an example of its usage:

```python
xp = hype.Experiment()
xp.sampleWhiteNoise()
xp.performAWT()
xp.findZeros()
xp.plotResults(boolShow=1) # plots the resulting signal and scalogram, and saves the figures
```
