import os
from pathlib import Path

import tomlkit

lock_path = Path("Cargo.lock")
lock = tomlkit.parse(lock_path.read_text())
for package in lock["package"]:
    if package["name"] in ("discrete-dists", "discrete-dists-core"):
        package["version"] = os.environ["CZ_PRE_NEW_VERSION"]
lock_path.write_text(tomlkit.dumps(lock))
