import os
from pathlib import Path

import tomlkit

PACKAGES = {
    "Cargo.lock": ("discrete-dists", "discrete-dists-core"),
    "uv.lock": ("discrete-dists",),
}

for lock_name, names in PACKAGES.items():
    lock_path = Path(lock_name)
    lock = tomlkit.parse(lock_path.read_text())
    for package in lock["package"]:
        if package["name"] in names:
            package["version"] = os.environ["CZ_PRE_NEW_VERSION"]
    lock_path.write_text(tomlkit.dumps(lock))
