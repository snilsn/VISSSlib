from VISSSlib.detection import *
import numpy as np

class TestDetection(object):

    def test_roi(self):

        img = np.random.random((100,100))

        for xr in range(0,40):
            for yr in range(0,40):
                roi = (xr,yr,40,40)
                imgE, xo, yo, _ = extractRoi(roi,img)
                
                imgE1, xo, yo, extraROI = extractRoi(roi, img, extra=20)
                imgE2, _,_,_ = extractRoi(extraROI, imgE1)

                assert np.all(imgE2==imgE), (xr, yr)
