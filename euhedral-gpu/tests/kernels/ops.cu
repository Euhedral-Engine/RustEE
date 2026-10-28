// The integration tests' kernels; ops.ptx is generated from this file (see its header).
extern "C" __global__ void axpb(float* y, const float* x, float a, float b, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + b; }
extern "C" __global__ void add_u32(unsigned* z, const unsigned* x, unsigned c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) z[i] = x[i] + c; }
extern "C" __global__ void spin(unsigned long long cycles) {
    long long t = clock64(); while ((unsigned long long)(clock64() - t) < cycles) {} }
extern "C" __global__ void poke(int* p) { *p = 1; }
