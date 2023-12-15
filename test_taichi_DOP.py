from utils import *
from crip.io import *
import numpy as np
from crip.mangoct import *
from os.path import abspath
import time
                             
                    
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
# cfg.setIO(abspath(f'./sgm'), abspath(f'./rec'), '.*', OutputFileReplace=['sgm_', 'rec_'])
cfg.setSgmConeBeam(SinogramWidth=sinogramWidth, SinogramHeight=sinogramHeight, Views=views,
                    DetectorElementSize=detectorElementSize, SliceCount=sliceCount,
                    SliceThickness=sliceThickness, SliceOffCenter=sliceOffCenter)
cfg.setRecConeBeam(imageDimension, pixelSize, imageSliceCount, imageSliceThickness, 'GaussianApodizedRamp', 0.5)
cfg.ShortScan = 0 if 360 - abs(scanAngle) < 0.01 else 1
sgm = imreadRaw('./sgm/sgm_water.raw', 360, 324, nSlice=237, dtype=np.float32)

# data-oriented programming(DOP)
start_time = time.time()

ti.init(arch=ti.cuda)
sgm_field = ti.field(dtype=ti.f32, shape=sgm.shape)
sgm_field.from_numpy(sgm)

sid_array = ti.field(dtype=ti.f32, shape=cfg.Views)
sdd_array = ti.field(dtype=ti.f32, shape=cfg.Views)
offcenter_array = ti.field(dtype=ti.f32, shape=cfg.Views)
init_distance_kernel(sid_array, cfg.SourceIsocenterDistance, cfg.Views)
init_distance_kernel(sdd_array, cfg.SourceDetectorDistance, cfg.Views)
init_distance_kernel(offcenter_array, cfg.DetectorOffcenter, cfg.Views)

u = ti.field(dtype=ti.f32, shape=cfg.SinogramWidth)
v = ti.field(dtype=ti.f32, shape=cfg.SliceCount)
init_u_kernel(u, cfg.SinogramWidth, cfg.DetectorElementSize, offcenter_array[0])
init_u_kernel(v, cfg.SliceCount, cfg.SliceThickness, cfg.SliceOffCenter)

beta = ti.field(dtype=ti.f32, shape=cfg.Views)
init_beta_kernel(beta, cfg.Views, cfg.ImageRotation, cfg.TotalScanAngle)

reconKernel = ti.field(dtype=ti.f32, shape=2*cfg.SinogramWidth-1)
reconKernel_ramp = ti.field(dtype=ti.f32, shape=2*cfg.SinogramWidth-1)
init_recon_kernel_gaussian_apodized(reconKernel, cfg.SinogramWidth, cfg.DetectorElementSize, cfg.GaussianApodizedRamp)
init_recon_kernel_hamming(reconKernel_ramp, cfg.SinogramWidth, cfg.DetectorElementSize, 1)

WeightSinogram_kernel(sgm_field, u, cfg.SinogramWidth, cfg.SinogramHeight, cfg.Views, cfg.SliceCount,
                      cfg.SliceThickness, cfg.SliceOffCenter, sdd_array, cfg.TotalScanAngle, cfg.ShortScan,
                      beta, offcenter_array)

sgm_field_flt_ramp = ti.field(dtype=ti.f32, shape=sgm.shape)
ConvolveSinogram_kernel(sgm_field_flt_ramp, sgm_field, reconKernel_ramp, cfg.SinogramWidth, cfg.SinogramHeight, 
                        cfg.Views, cfg.SliceCount, u, cfg.DetectorElementSize)
sgm_field_flt = ti.field(dtype=ti.f32, shape=sgm.shape)
ConvolveSinogram_kernel(sgm_field_flt, sgm_field_flt_ramp, reconKernel, cfg.SinogramWidth, cfg.SinogramHeight, 
                        cfg.Views, cfg.SliceCount, u, cfg.DetectorElementSize)


img = ti.field(dtype=ti.f32, shape=(cfg.ImageSliceCount, cfg.ImageDimension, cfg.ImageDimension))
backproject_pixel_driven_kernel(sgm_field_flt, img, u, v, beta, cfg.ShortScan, cfg.SinogramWidth, cfg.Views, 
                                cfg.SliceCount, int(cfg.ConeBeam), cfg.ImageDimension, cfg.ImageSliceCount,
                                sdd_array, sid_array, offcenter_array, cfg.PixelSize, cfg.ImageSliceThickness, 
                                cfg.ImageCenter[0], cfg.ImageCenter[1], cfg.ImageCenterZ)
img = img.to_numpy()
print('Time elapsed: {:.2f} s'.format(time.time() - start_time))

imwriteTiff(img, './rec/rec_water.tiff')
