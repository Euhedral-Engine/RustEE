// The acceptance example's own kernels. Integer arithmetic keeps the CPU reference exact.

// y[i] = x[i] * a + i
extern "C" __global__ void mix(unsigned* y, const unsigned* x, unsigned a, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * a + i;
}

// z[i] = (y[i] ^ b) + (y[i] >> 3)
extern "C" __global__ void fold(unsigned* z, const unsigned* y, unsigned b, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = (y[i] ^ b) + (y[i] >> 3);
}
