#!/usr/bin/env python3

import packaging.version
import sys
import os
import json
import re

URL = "https://xsuite.github.io/xplt/{}/"

if __name__ == "__main__":

    # Collect versions
    versions = []
    pre_releases = []
    for dir in os.listdir():
        if re.match(r"^\d+(\.\d+)*$", dir):
            versions.append(dir)
            if os.path.exists(os.path.join(dir, ".pre-release")):
                pre_releases.append(dir)
    versions.sort(key=packaging.version.parse, reverse=True)
    latest = list(filter(lambda v: v not in pre_releases, versions))[0]

    # Generate versions file
    versions_json = []
    for ver in versions:
        versions_json.append(
            {
                "name": f"{ver}" + (" (latest)" if ver == latest else " (pre-release)" if ver in pre_releases else ""),
                "version": ver,
                "url": URL.format(ver),
            }
        )
    with open("versions.json", "w") as f:
        json.dump(versions_json, f, indent=True)

    # generate redirect
    with open("index.html", "w") as f:
        f.write(f"""<meta http-equiv="refresh" content="0; url={URL.format('latest')}">\n""")

    # update symlink
    if os.path.exists("latest"):
        os.remove("latest")
    os.symlink(latest, "latest")
