
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Error: Missing required argument.")
    print("Usage: python verifier.py <dataset-path>")
    sys.exit(1)

p = sys.argv[1]
a = np.memmap(p, dtype="<u2", mode="r")
n = a.shape[0]
assert a.nbytes % 2 == 0, "byte count not even — truncated write"
print(f"{n:,} tokens  ({a.nbytes/1e9:.1f} GB)")
print("last token:", int(a[-1]), " (expect 65535, the doc separator)")

SEP, seps, mx = 0xFFFF, 0, 0
step = 1 << 28                       # 256M elems/chunk, memory-safe
for i in range(0, n, step):
    c = np.asarray(a[i:i+step])
    seps += int(np.count_nonzero(c == SEP))
    real = c[c != SEP]
    if real.size:
        mx = max(mx, int(real.max()))
print(f"documents (separators): {seps:,}")
print(f"content tokens:         {n - seps:,}")
print(f"max non-separator id:   {mx}   (must be < 50280 for OLMo)")
print(f"avg tokens/doc:         {(n - seps)/seps:.1f}")
