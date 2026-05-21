"""Batch-fix non-ASCII bytes in all .py files for cp1252 compatibility."""
import os

root = r'e:\crypto_trade'
files_fixed = 0

# Replacement map for known Unicode -> ASCII
REPLACEMENTS = [
    (b'\xe2\x80\x94', b'--'),    # em-dash
    (b'\xe2\x80\x93', b'-'),     # en-dash
    (b'\xe2\x86\x92', b'->'),    # right arrow
    (b'\xe2\x86\x90', b'<-'),    # left arrow
    (b'\xe2\x96\xba', b'>'),     # right pointer
    (b'\xe2\x80\x99', b"'"),     # right single quote
    (b'\xe2\x80\x98', b"'"),     # left single quote
    (b'\xe2\x80\x9c', b'"'),     # left double quote
    (b'\xe2\x80\x9d', b'"'),     # right double quote
    (b'\xc3\x97', b'x'),         # multiplication sign
    (b'\xe2\x89\xa5', b'>='),    # >=
    (b'\xe2\x89\xa4', b'<='),    # <=
    (b'\xe2\x88\x92', b'-'),     # minus sign
    (b'\xc2\xb7', b'*'),         # middle dot
    (b'\xc2\xb1', b'+/-'),       # plus-minus
]

# Box drawing characters (U+2500 to U+257F)
for c in range(0x2500, 0x2580):
    utf8 = chr(c).encode('utf-8')
    REPLACEMENTS.append((utf8, b'-'))

# Double-line box drawing (U+2550 to U+256C)
for c in range(0x2550, 0x256D):
    utf8 = chr(c).encode('utf-8')
    REPLACEMENTS.append((utf8, b'='))


def fix_file(path):
    with open(path, 'rb') as f:
        data = f.read()

    original_bad = sum(1 for b in data if b > 127)
    if original_bad == 0:
        return False

    for old, new in REPLACEMENTS:
        data = data.replace(old, new)

    remaining = sum(1 for b in data if b > 127)

    with open(path, 'wb') as f:
        f.write(data)

    status = "CLEAN" if remaining == 0 else f"{remaining} remaining"
    print(f"  Fixed {os.path.relpath(path, root)}: {original_bad} -> {status}")
    return True


# Scan all .py files in src/ and config/
scan_dirs = [
    os.path.join(root, 'src'),
    os.path.join(root, 'config'),
]

for scan_dir in scan_dirs:
    for dirpath, dirs, files in os.walk(scan_dir):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(dirpath, f)
                if fix_file(path):
                    files_fixed += 1

# Also fix root-level scripts
for f in os.listdir(root):
    if f.endswith('.py'):
        path = os.path.join(root, f)
        if fix_file(path):
            files_fixed += 1

print(f"\nDone! Fixed {files_fixed} files.")
