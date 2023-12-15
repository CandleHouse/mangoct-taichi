import taichi as ti


PI = 3.1415926536


@ti.kernel
def init_distance_kernel(distance_array: ti.template(), distance: ti.f32, V: ti.i32):
    for tid in range(V):
        distance_array[tid] = distance


@ti.kernel
def init_u_kernel(u: ti.template(), N: ti.i32, du: ti.f32, offcenter: ti.f32):
    for tid in range(N):
        u[tid] = (tid - (N - 1) / 2.0) * du + offcenter
        
        
@ti.kernel
def init_beta_kernel(beta: ti.template(), V: ti.i32, rotation: ti.f32, total_scan_angle: ti.f32):
    for tid in range(V):
        beta[tid] = (total_scan_angle / V * tid + rotation) * PI / 180


@ti.kernel
def init_recon_kernel_hamming(recon_kernel: ti.template(), N: ti.i32, du: ti.f32, t: ti.f32):
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
def init_recon_kernel_gaussian_apodized(recon_kernel: ti.template(), N: ti.i32, du: ti.f32, delta: ti.f32):
    if True:
        temp_sum = 0.0
        for i in range(2 * N - 1):
            n = i - (N - 1)
            recon_kernel[i] = ti.exp(-n * n / 2.0 / delta / delta)
            temp_sum += recon_kernel[i]
        
        for i in range(2 * N - 1):
            recon_kernel[i] = recon_kernel[i] / temp_sum / du
        
        
@ti.kernel
def WeightSinogram_kernel(sgm: ti.template(), u: ti.template(), N: ti.int32, H: ti.int32, V: ti.int32,
                          S: ti.int32, sliceThickness: ti.f32, sliceOffcenter: ti.f32,
                          sdd_array: ti.template(), totalScanAngle: ti.f32, shortScan: ti.int32,
                          beta_array: ti.template(), offcenter_array: ti.template()):
    for col, row in ti.ndrange(N, V):
        offcenter_bias = offcenter_array[row] - offcenter_array[0]
        u_actual = u[col] + offcenter_bias

        sdd = sdd_array[row]

        for i in range(S):
            v = sliceThickness * (i - S / 2.0 + 0.5) + sliceOffcenter
            sgm[i, row, col] *= sdd * sdd / ti.sqrt(u_actual * u_actual + sdd * sdd + v * v)

        if shortScan:
            beta = abs(beta_array[row] - beta_array[0])
            rotation_direction = abs(totalScanAngle) / (totalScanAngle)
            gamma = ti.atan2(u_actual, sdd) * rotation_direction

            gamma_max = abs(totalScanAngle) * PI / 180.0 - PI

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
def ConvolveSinogram_kernel(sgm_flt: ti.template(), sgm: ti.template(), reconKernel: ti.template(),
                            N: ti.int32, H: ti.int32, V: ti.int32, S: ti.int32, u: ti.template(), du: ti.f32):
    for col, row in ti.ndrange(N, V):
        for slice in range(S):
            sgm_flt_local = 0.0

            for i in range(N):
                sgm_flt_local += sgm[slice, row, i] * reconKernel[N - 1 - col + i]

            sgm_flt[slice, row, col] = sgm_flt_local * du


@ti.kernel
def backproject_pixel_driven_kernel(sgm: ti.template(), img: ti.template(), u: ti.template(), v: ti.template(), beta: ti.template(),
                                    short_scan: ti.template(), N: ti.template(), V: ti.template(), S: ti.template(),
                                    cone_beam: ti.i32, M: ti.template(), imgS: ti.template(),
                                    sdd_array: ti.template(), sid_array: ti.template(), offcenter_array: ti.template(),
                                    dx: ti.template(), dz: ti.template(), xc: ti.template(), yc: ti.template(), zc: ti.template()):
    for col, row, imgS_idx in ti.ndrange(M, M, imgS):
        du = u[1] - u[0]
        dv = v[1] - v[0]

        if imgS_idx <= imgS:
            x = (col - (M - 1) / 2.0) * dx + xc
            y = ((M - 1) / 2.0 - row) * dx + yc
            delta_beta = 0.0
            
            for slice in range(imgS_idx, imgS_idx + 1):
                z = (slice - (float(imgS) - 1.0) / 2.0) * dz + zc

                img_local = 0.0

                for view in range(V):
                    offcenter_bias = offcenter_array[view] - offcenter_array[0]
                    sid = sid_array[view]
                    sdd = sdd_array[view]

                    if view == 0:
                        delta_beta = ti.abs(beta[1] - beta[0])
                    elif view == V - 1:
                        delta_beta = ti.abs(beta[view] - beta[view - 1])
                    else:
                        delta_beta = ti.abs(beta[view + 1] - beta[view - 1]) / 2.0

                    U = sid - x * ti.cos(beta[view]) - y * ti.sin(beta[view])
                    mag_factor = sdd / U
                    u0 = mag_factor * (x * ti.sin(beta[view]) - y * ti.cos(beta[view]))

                    k = ti.cast(ti.floor((u0 - (u[0] + offcenter_bias)) / du), ti.i32)
                    if k < 0 or k + 1 > N - 1:
                        img_local = 0
                        break

                    w = (u0 - (u[k] + offcenter_bias)) / du

                    if cone_beam and ti.abs(dv) > 0.00001:
                        v0 = mag_factor * z
                        k_z = ti.cast(ti.floor((v0 - v[0]) / dv), ti.i32)
                        if k_z < 0 or k_z + 1 > S - 1:
                            img_local = 0
                            break

                        w_z = (v0 - v[k_z]) / dv

                        lower_row_val = (w * sgm[k_z, view, k+1] + (1 - w) * sgm[k_z, view, k])
                        upper_row_val = (w * sgm[k_z+1, view, k+1] + (1 - w) * sgm[k_z+1, view, k])

                        img_local += sid / U / U * (w_z * upper_row_val + (1 - w_z) * lower_row_val) * delta_beta
                    else:
                        img_local += sid / U / U * (w * sgm[slice, view, k+1] + (1 - w) * sgm[slice, view, k]) * delta_beta

                if short_scan:
                    img[imgS_idx, row, col] = img_local
                else:
                    img[imgS_idx, row, col] = img_local / 2.0
