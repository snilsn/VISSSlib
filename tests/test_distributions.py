from VISSSlib.distributions import *
import numpy as np

nSample = 100
seed = 0

class TestVolume(object):

    def test_VolumeInterpolation(self):
        width = 1280
        height = 1024

        np.random.seed(seed)
        phi, theta, Of_z = np.random.random(3)
        minDmax, maxDmax = 0,20
        sizeBins = np.linspace(minDmax, maxDmax)
        D_highRes, V_highRes = estimateVolumes(width, height, phi, theta, Of_z, sizeBins, nSteps=21, interpolate=False)
        D_lowRes, V_lowRes = estimateVolumes(width, height, phi, theta, Of_z, sizeBins, nSteps=2, interpolate=False)
        D_lowResInterp, V_lowResInterp = interpolateVolumes(sizeBins, D_lowRes, V_lowRes)

        assert np.allclose(V_lowResInterp,  V_highRes)

    def test_volumeEstimate(self):
        width = 1280
        height = 1024

        #no rotation!
        phi, theta, Of_z = 0,0,0

        V = estimateVolume(width, height, phi, theta, Of_z)

        assert np.isclose(V, width*width*height)



