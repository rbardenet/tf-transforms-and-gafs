import pytest
import sys
import os
import glob
sys.path.append("hyperbolic")
import hyperbolic as hype

def test_Experiment():
    """
    make sure the main experiment runs smoothly and produces plots
    """

    # Run experiment
    xp = hype.Experiment(alpha=2.0, expId="pytest")
    xp.sampleWhiteNoise()
    xp.performAWT()
    xp.findZeros()
    xp.plotResults(boolShow=0)

    # Check wheter 3 plots have been created
    pattern = "*_pytest_alpha*.pdf"
    figNames = glob.glob(pattern)
    os.popen("rm "+pattern)
    print(figNames)
    print(len(figNames))
    assert(len(figNames)==3)
