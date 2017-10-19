import sys
sys.path.append("../hyperbolic")
import hyperbolic as hype

xp = hype.Experiment(alpha=2.0) # initialize experiment, you can play with the parameters, hit shift+TAB for details
xp.sampleWhiteNoise()
xp.performAWT()
xp.findZeros()
xp.plotResults(boolShow=0)
