import tvm
from tvm import te
m = te.var(name='m')
A = te.placeholder((m,), dtype='float32', name='a')
B = te.placeholder((m,), dtype='float32', name='b')
C = te.compute((m,), lambda i: A[i] + B[i], name='c')
S = te.create_schedule(C.op)
print(type(A), type(B), type(C))