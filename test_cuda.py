from crip.physics import *
from crip.mangoct import *
from os.path import abspath
from crip.io import *
from crip.preprocess import *
import numpy as np


# System settings.
SID = 750
SDD = 1250
sinogramWidth = 324
sinogramHeight = 360
views = 360
detectorElementSize = 1.232  # mm
sliceCount = 237
sliceThickness = 1.232  # mm
sliceOffCenter = 0  # mm
imageDimension = 256
pixelSize = 1  # mm
imageSliceCount = 60
imageSliceThickness = 2.87  # mm
scanAngle = -360

cfg = MgfbpConfig()
cfg.setGeometry(SID, SDD, scanAngle)
cfg.setIO(abspath(f'./sgm'), abspath(f'./rec'), '.*', OutputFileReplace=['sgm_', 'rec_'])
cfg.setSgmConeBeam(SinogramWidth=sinogramWidth, SinogramHeight=sinogramHeight, Views=views,
                    DetectorElementSize=detectorElementSize, SliceCount=sliceCount,
                    SliceThickness=sliceThickness, SliceOffCenter=sliceOffCenter)
cfg.setRecConeBeam(imageDimension, pixelSize, imageSliceCount, imageSliceThickness, 'GaussianApodizedRamp', 0.5)
Mgfbp(exe='mgfbp', cudaDevice='0').exec(cfg)
