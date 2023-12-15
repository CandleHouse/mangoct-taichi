import taichi as ti
from crip.mangoct import *
import numpy as np


PI = 3.1415926536


@ti.data_oriented
class Mgfbp_Taichi:
    def __init__(self, cfg: MgfbpConfig):
        self.cfg = cfg
        self.cfg.ShortScan = 0 if 360 - abs(cfg.TotalScanAngle) < 0.01 else 1
        ti.init(arch=ti.cuda)

        self.sid_array = ti.field(dtype=ti.f32, shape=self.cfg.Views)
        self.sdd_array = ti.field(dtype=ti.f32, shape=self.cfg.Views)
        self.offcenter_array = ti.field(dtype=ti.f32, shape=self.cfg.Views)
        self.u = ti.field(dtype=ti.f32, shape=self.cfg.SinogramWidth)
        self.v = ti.field(dtype=ti.f32, shape=self.cfg.SliceCount)
        self.beta = ti.field(dtype=ti.f32, shape=self.cfg.Views)
        self.reconKernel = ti.field(dtype=ti.f32, shape=2*self.cfg.SinogramWidth-1)
        self.reconKernel_ramp = ti.field(dtype=ti.f32, shape=2*self.cfg.SinogramWidth-1)
        self.img = ti.field(dtype=ti.f32, shape=(self.cfg.ImageSliceCount, 
                            self.cfg.ImageDimension, self.cfg.ImageDimension))
        self.init_param()
    
    def fdk(self, sgm: np.ndarray):
        sgm_field = ti.field(dtype=ti.f32, shape=sgm.shape)
        sgm_field.from_numpy(sgm)
        
        sgm_field_flt_ramp = ti.field(dtype=ti.f32, shape=sgm.shape)
        sgm_field_flt = ti.field(dtype=ti.f32, shape=sgm.shape)
        img = ti.field(dtype=ti.f32, shape=(self.cfg.ImageSliceCount, self.cfg.ImageDimension, self.cfg.ImageDimension))
        
        self.WeightSinogram_kernel(sgm_field)
        self.ConvolveSinogram_kernel(sgm_field_flt_ramp, sgm_field, self.reconKernel_ramp)
        self.ConvolveSinogram_kernel(sgm_field_flt, sgm_field_flt_ramp, self.reconKernel)
        self.Backproject_pixel_driven_kernel(sgm_field_flt, img)
        
        return img.to_numpy()
        
    def init_param(self):
        self.init_distance_kernel(self.sid_array, self.cfg.SourceIsocenterDistance, self.cfg.Views)
        self.init_distance_kernel(self.sdd_array, self.cfg.SourceDetectorDistance, self.cfg.Views)
        self.init_distance_kernel(self.offcenter_array, self.cfg.DetectorOffcenter, self.cfg.Views)
        self.init_u_kernel(self.u, self.cfg.SinogramWidth, self.cfg.DetectorElementSize, self.offcenter_array[0])
        self.init_u_kernel(self.v, self.cfg.SliceCount, self.cfg.SliceThickness, self.cfg.SliceOffCenter)
        self.init_beta_kernel(self.beta, self.cfg.Views, self.cfg.ImageRotation, self.cfg.TotalScanAngle)
        self.init_recon_kernel_gaussian_apodized(self.reconKernel, self.cfg.SinogramWidth, 
                                                 self.cfg.DetectorElementSize, self.cfg.GaussianApodizedRamp)
        self.init_recon_kernel_hamming(self.reconKernel_ramp, self.cfg.SinogramWidth, self.cfg.DetectorElementSize, 1)
        
    @ti.kernel
    def init_distance_kernel(self, distance_array: ti.template(), distance: ti.f32, V: ti.i32):
        for tid in range(V):
            distance_array[tid] = distance

    @ti.kernel
    def init_u_kernel(self, u: ti.template(), N: ti.i32, du: ti.f32, offcenter: ti.f32):
        for tid in range(N):
            u[tid] = (tid - (N - 1) / 2.0) * du + offcenter
            
    @ti.kernel
    def init_beta_kernel(self, beta: ti.template(), V: ti.i32, rotation: ti.f32, total_scan_angle: ti.f32):
        for tid in range(V):
            beta[tid] = (total_scan_angle / V * tid + rotation) * PI / 180

    @ti.kernel
    def init_recon_kernel_hamming(self, recon_kernel: ti.template(), N: ti.i32, du: ti.f32, t: ti.f32):
        for tid in range(2 * N - 1):
            n = tid - (N - 1)

            # ramp part
            if n == 0:
                recon_kernel[tid] = t / (4 * du * du)
            elif n % 2 == 0:
                recon_kernel[tid] = 0
            else:
                recon_kernel[tid] = -t / (n * n * PI * PI * du * du)

            # cosine part
            sgn = 1 if n % 2 == 0 else -1

            recon_kernel[tid] += (1 - t) * (sgn / (2 * PI * du * du) * (1.0 / (1 + 2 * n) + 1.0 / (1 - 2 * n))
                - 1 / (PI * PI * du * du) * (1.0 / (1 + 2 * n) / (1 + 2 * n) + 1.0 / (1 - 2 * n) / (1 - 2 * n)))

    @ti.kernel
    def init_recon_kernel_gaussian_apodized(self, recon_kernel: ti.template(), N: ti.i32, du: ti.f32, delta: ti.f32):
        if True:
            temp_sum = 0.0
            for i in range(2 * N - 1):
                n = i - (N - 1)
                recon_kernel[i] = ti.exp(-n * n / 2.0 / delta / delta)
                temp_sum += recon_kernel[i]
            
            for i in range(2 * N - 1):
                recon_kernel[i] = recon_kernel[i] / temp_sum / du
                
                
    @ti.kernel
    def WeightSinogram_kernel(self, sgm: ti.template()):
        N = self.cfg.SinogramWidth
        H = self.cfg.SinogramHeight
        V = self.cfg.Views
        S = self.cfg.SliceCount
        
        for col, row in ti.ndrange(N, V):
            offcenter_bias = self.offcenter_array[row] - self.offcenter_array[0]
            u_actual = self.u[col] + offcenter_bias

            sdd = self.sdd_array[row]

            for i in range(S):
                v = self.cfg.SliceThickness * (i - S / 2.0 + 0.5) + self.cfg.SliceOffCenter
                sgm[i, row, col] *= sdd * sdd / ti.sqrt(u_actual * u_actual + sdd * sdd + v * v)

            if self.cfg.ShortScan:
                beta = abs(self.beta[row] - self.beta[0])
                rotation_direction = abs(self.cfg.TotalScanAngle) / (self.cfg.TotalScanAngle)
                gamma = ti.atan2(u_actual, sdd) * rotation_direction

                gamma_max = abs(self.cfg.TotalScanAngle) * PI / 180.0 - PI

                weighting = 0.0
                if 0 <= beta < gamma_max - 2 * gamma:
                    weighting = ti.sin(PI / 2 * beta / (gamma_max - 2 * gamma))**2
                elif gamma_max - 2 * gamma <= beta < PI - 2 * gamma:
                    weighting = 1.0
                elif PI - 2 * gamma <= beta <= PI + gamma_max:
                    weighting = ti.sin(PI / 2 * (PI + gamma_max - beta) / (gamma_max + 2 * gamma))**2

                for i in range(S):
                    sgm[i, row, col] *= weighting
            else:
                pass


    @ti.kernel
    def ConvolveSinogram_kernel(self, sgm_flt: ti.template(), sgm: ti.template(), reconKernel: ti.template()):
        N = self.cfg.SinogramWidth
        H = self.cfg.SinogramHeight
        V = self.cfg.Views
        S = self.cfg.SliceCount
        
        for col, row in ti.ndrange(N, V):
            for slice in range(S):
                sgm_flt_local = 0.0

                for i in range(N):
                    sgm_flt_local += sgm[slice, row, i] * reconKernel[N - 1 - col + i]

                sgm_flt[slice, row, col] = sgm_flt_local * self.cfg.DetectorElementSize


    @ti.kernel
    def Backproject_pixel_driven_kernel(self, sgm: ti.template(), img: ti.template()):
        N = self.cfg.SinogramWidth
        H = self.cfg.SinogramHeight
        V = self.cfg.Views
        S = self.cfg.SliceCount
        M = self.cfg.ImageDimension
        imgS = self.cfg.ImageSliceCount
        dx = self.cfg.PixelSize
        dz = self.cfg.ImageSliceThickness
        xc = self.cfg.ImageCenter[0]
        yc = self.cfg.ImageCenter[1]
        zc = self.cfg.ImageCenterZ
        
        for col, row, imgS_idx in ti.ndrange(M, M, imgS):
            du = self.u[1] - self.u[0]
            dv = self.v[1] - self.v[0]

            if imgS_idx <= imgS:
                x = (col - (M - 1) / 2.0) * dx + xc
                y = ((M - 1) / 2.0 - row) * dx + yc
                delta_beta = 0.0
                
                for slice in range(imgS_idx, imgS_idx + 1):
                    z = (slice - (float(imgS) - 1.0) / 2.0) * dz + zc

                    img_local = 0.0

                    for view in range(V):
                        offcenter_bias = self.offcenter_array[view] - self.offcenter_array[0]
                        sid = self.sid_array[view]
                        sdd = self.sdd_array[view]

                        if view == 0:
                            delta_beta = ti.abs(self.beta[1] - self.beta[0])
                        elif view == V - 1:
                            delta_beta = ti.abs(self.beta[view] - self.beta[view - 1])
                        else:
                            delta_beta = ti.abs(self.beta[view + 1] - self.beta[view - 1]) / 2.0

                        U = sid - x * ti.cos(self.beta[view]) - y * ti.sin(self.beta[view])
                        mag_factor = sdd / U
                        u0 = mag_factor * (x * ti.sin(self.beta[view]) - y * ti.cos(self.beta[view]))

                        k = ti.cast(ti.floor((u0 - (self.u[0] + offcenter_bias)) / du), ti.i32)
                        if k < 0 or k + 1 > N - 1:
                            img_local = 0
                            break

                        w = (u0 - (self.u[k] + offcenter_bias)) / du

                        if self.cfg.ConeBeam and ti.abs(dv) > 0.00001:
                            v0 = mag_factor * z
                            k_z = ti.cast(ti.floor((v0 - self.v[0]) / dv), ti.i32)
                            if k_z < 0 or k_z + 1 > S - 1:
                                img_local = 0
                                break

                            w_z = (v0 - self.v[k_z]) / dv

                            lower_row_val = (w * sgm[k_z, view, k+1] + (1 - w) * sgm[k_z, view, k])
                            upper_row_val = (w * sgm[k_z+1, view, k+1] + (1 - w) * sgm[k_z+1, view, k])

                            img_local += sid / U / U * (w_z * upper_row_val + (1 - w_z) * lower_row_val) * delta_beta
                        else:
                            img_local += sid / U / U * (w * sgm[slice, view, k+1] + (1 - w) * sgm[slice, view, k]) * delta_beta

                    if self.cfg.ShortScan:
                        img[imgS_idx, row, col] = img_local
                    else:
                        img[imgS_idx, row, col] = img_local / 2.0
