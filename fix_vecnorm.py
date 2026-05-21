"""Fix VecNormalize pickle for numpy 1.x / 2.x cross-compat.

The pickle was saved with numpy 2.x which stores PCG64 class objects
directly. numpy 1.x expects string names. We do a byte-level fix.
"""
import re
import pickle
import numpy as np

path = r'E:\crypto_trade\models_saved\vecnorm_ppo_v4_final.pkl'

print(f"Loading {path}...")

with open(path, 'rb') as f:
    data = f.read()

# The problem: numpy 2.x pickle stores the PCG64 class object reference.
# numpy 1.x __bit_generator_ctor expects a string name.
# We can fix this by patching the numpy _pickle module at runtime.

original_ctor = np.random._pickle.__bit_generator_ctor

def fixed_ctor(bit_generator_name):
    """Accept both class objects and string names."""
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    return original_ctor(bit_generator_name)

# Monkey-patch
np.random._pickle.__bit_generator_ctor = fixed_ctor

# Also handle numpy._core -> numpy.core module renames
class Compat2to1Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)

# Now handle the state format issue - numpy 2.x PCG64 state has 'bit_generator'
# as a class, not a string. We need to patch PCG64's state setter.
# Since we can't monkeypatch Cython, we'll intercept during unpickling.

# Use a wrapper approach
class PCG64Wrapper:
    """Wrapper that creates a real PCG64 but handles state compat."""
    def __reduce__(self):
        return (np.random._pickle.__bit_generator_ctor, ('PCG64',))

    def __setstate__(self, state):
        # Convert numpy 2.x state format to 1.x
        if isinstance(state, dict):
            bg_name = state.get('bit_generator', 'PCG64')
            if isinstance(bg_name, type):
                state['bit_generator'] = bg_name.__name__
            # The 's' key in 'state' sub-dict might be stored differently
            inner = state.get('state', {})
            if isinstance(inner.get('state', None), int):
                # numpy 2.x stores state as int, 1.x stores as dict
                pass
        self._pcg64 = np.random.PCG64()
        try:
            self._pcg64.state = state
        except (TypeError, ValueError):
            print("  [WARN] Could not restore RNG state, using fresh PCG64")

# Actually, let's try the simplest approach - just re-create the pickle
# by loading with compat and catching the state error

import io

class SkipBadStateUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)

    def load_reduce(self):
        try:
            return super().load_reduce()
        except (TypeError, ValueError) as e:
            # If it's the PCG64 issue, push a fresh PCG64 onto the stack
            print(f"  [WARN] Caught error during reduce: {e}")
            self.stack[-1] = np.random.PCG64()

# Try yet another approach: just replace the bytes
# In the pickle, PCG64 is stored as:
# \x8c\x16numpy.random._pcg64 \x8c\x05PCG64 \x93 (STACK_GLOBAL)
# followed by the state via BUILD
# The __bit_generator_ctor is called with the CLASS as arg

# The fix: just load it with the patched ctor and catch the state error
from unittest.mock import patch
import numpy.random._pcg64 as pcg64_mod

# Save original
orig_pcg64_setstate = pcg64_mod.PCG64.__setstate__

# We can't patch Cython __setstate__, so let's use a different strategy:
# Wrap the entire pickle load in a custom unpickler that intercepts
# the BUILD opcode for PCG64 objects

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)

    def load_build(self):
        # Peek at stack - if top is state dict for a PCG64, handle it
        stack = self.stack
        state = stack[-1]
        inst = stack[-2] if len(stack) >= 2 else None

        if isinstance(inst, np.random.PCG64) and isinstance(state, dict):
            # Convert class references to strings in state
            if 'bit_generator' in state:
                bg = state['bit_generator']
                if isinstance(bg, type):
                    state['bit_generator'] = bg.__name__
            # Try to set state, fall back to fresh
            try:
                inst.__setstate__(state)
            except (TypeError, ValueError):
                print("  [WARN] Incompatible PCG64 state format, using fresh RNG")
            stack.pop()  # Remove state from stack
            return

        # Default behavior
        super().load_build()

    # Register the opcode handler
    pickle.Unpickler.dispatch = dict(pickle.Unpickler.dispatch)


# Actually dispatch is trickier. Let me just do a direct byte patch.
print("Attempting byte-level pickle fix...")

# Replace 'numpy._core.multiarray' -> 'numpy.core.multiarray' in the bytes
fixed_data = data.replace(b'numpy._core.multiarray', b'numpy.core.multiarray')
fixed_data = fixed_data.replace(b'numpy._core.numeric', b'numpy.core.numeric\x00')

# For the PCG64 issue, replace the class reference pattern
# The ctor receives a class object. We need it to receive a string.
# Actually, let's just load with the patched ctor
with open(path + '.bak', 'wb') as f:
    f.write(data)
print("Backup saved as .bak")

try:
    obj = pickle.loads(fixed_data)
    print(f"Loaded! Type: {type(obj)}")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("Re-saved successfully!")
except Exception as e:
    print(f"Byte fix failed: {e}")
    print("Trying with patched ctor only...")
    try:
        obj = pickle.loads(data)
        print(f"Loaded with patched ctor! Type: {type(obj)}")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print("Re-saved successfully!")
    except Exception as e2:
        print(f"Also failed: {e2}")
        print("The model needs numpy 2.x. Consider: pip install 'numpy>=2.0'")
