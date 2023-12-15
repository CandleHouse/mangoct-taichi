# Mangoct Taichi: Taichi enabled cross platform CT Image Projection and Reconstruction Program (Demo)

## Whats new?

- Now can use only '.py' files to do medical image reconstruction with the help of [Taichi](https://github.com/taichi-dev/taichi), without to take care of compile.

  - [x] Cone beam reconstruction support (only)
  - [x] NO OTHER THINGS SUPPORT!
- Spending almost the same amount of time as using pure cuda programs.

## For what use?

- **Test driven development** reconstruction program

## Prerequisites

- [taichi](https://github.com/taichi-dev/taichi)
- [crip](https://github.com/SEU-CT-Recon/crip)
- [mangoct series](https://github.com/SEU-CT-Recon/mandoct) if possible for `./test_cuda.py`

## Files

```
./test_cuda.py        # use raw mangoct cuda program, the fastest speed
./test_taichi_DOP.py  # taichi enabled reconstruction
./test_taichi_ODOP.py # taichi enabled reconstruction (elegant)
```

