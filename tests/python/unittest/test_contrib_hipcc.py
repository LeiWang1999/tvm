from tvm.contrib.hipcc import compile_hip

# Example usage
hip_code = """
__global__ void add(float *a, float *b, float *c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
"""

arch = "gfx90a"  # Replace with your AMD GPU architecture
hsaco = compile_hip(hip_code, arch=arch)
