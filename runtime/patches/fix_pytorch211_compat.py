#!/usr/bin/env python3
"""
Patch vLLM for PyTorch 2.11 (NGC 26.03) compatibility.

Two fixes:
1. Remove hoist=True from register_opaque_type() — kwarg removed in PyTorch 2.11
2. Fix __fx_repr__ return type: {ModuleName} (set) → {"ModuleName": ModuleName} (dict)
   PyTorch 2.11's get_opaque_obj_repr() expects dict[str, type], not set
"""
import glob
import sys

target = None
for pattern in [
    "/workspace/vllm-src/vllm/utils/torch_utils.py",
    "/tmp/vllm/vllm/utils/torch_utils.py",
    "/usr/local/lib/python3.*/dist-packages/vllm/utils/torch_utils.py",
]:
    for p in glob.glob(pattern):
        target = p
        break
    if target:
        break

if not target:
    print("FATAL: torch_utils.py not found")
    sys.exit(1)

with open(target) as f:
    code = f.read()

changes = 0

# Fix 1: hoist=True
if ", hoist=True" in code:
    code = code.replace(", hoist=True", "")
    changes += 1
    print(f"[fix1] Removed hoist=True")
elif "hoist=True" not in code:
    print(f"[fix1] hoist=True already absent — OK")
else:
    print(f"FATAL: hoist=True in unexpected position")
    sys.exit(1)

# Fix 2: __fx_repr__ set → dict
# Before: return (f"ModuleName({self.value!r})", {ModuleName})
# After:  return (f"ModuleName({self.value!r})", {"ModuleName": ModuleName})
if "{ModuleName}" in code:
    code = code.replace("{ModuleName}", '{"ModuleName": ModuleName}')
    changes += 1
    print(f"[fix2] Fixed __fx_repr__ set → dict")
elif '{"ModuleName": ModuleName}' in code:
    print(f"[fix2] Already patched — OK")
else:
    print(f"WARNING: {'{ModuleName}'} pattern not found — may need manual check")

if changes > 0:
    with open(target, "w") as f:
        f.write(code)
    print(f"PATCHED: {target} ({changes} changes)")
else:
    print(f"NO CHANGES needed: {target}")

# Verify
with open(target) as f:
    final = f.read()
assert "hoist=True" not in final, "FATAL: hoist=True still present"
assert '{ModuleName}' not in final or '{"ModuleName": ModuleName}' in final, "FATAL: set pattern still present"
print("VERIFIED: All patches applied correctly")
