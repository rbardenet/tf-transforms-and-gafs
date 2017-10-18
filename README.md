## hyperbolic

This project is the companion code to the paper XXX.

[![CheckHowThisButtonIsDoneToAdaptItToYourProject](https://travis-ci.org/[your_username]/[project_name].svg?branch=[branch_to_test])](https://travis-ci.org/[your_username]/[project_name])


> project description

_hyperbolic_ is a Python 3 library implementing Paul's continuous wavelet transform of an analytic white Gaussian noise, see XXX for details.

### Install from sources

Clone (or download) this repository

```bash
git clone https://github.com/rbardenet/hyperbolic.git
cd hyperbolic
```

and execute `setup.py`

```bash
pip install .
```

If you're in development mode and you want to install also dev packages, documentation and/or tests, you can do as follows:

```bash
pip install -e .
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
