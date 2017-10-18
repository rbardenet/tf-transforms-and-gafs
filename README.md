## [hyperbolic]

This project is the companion code to the paper XXX.

[![CheckHowThisButtonIsDoneToAdaptItToYourProject](https://travis-ci.org/[your_username]/[project_name].svg?branch=[branch_to_test])](https://travis-ci.org/[your_username]/[project_name])


> project description

[hyperbolic] is a Python 3 library implementing the analytic wavelet transform of a white Gaussian noise on the Hardy space $H^2$.

### Install from sources

Clone (or download) this repository

```bash
git clone https://github.com/[myself]/[project_name].git
cd [project_name]
```

And execute `setup.py`

```bash
pip install .
```

If you're in development mode and you want to install also dev packages, documentation and/or tests, you can do as follows:

```bash
pip install -e .
```

## Usage examples

You can import [hyperbolic] by doing

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
