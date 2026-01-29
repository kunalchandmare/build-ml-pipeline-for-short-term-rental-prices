import yaml
from pathlib import Path
from packaging.version import Version, InvalidVersion
from collections import defaultdict
import re

def parse_dep(dep_str: str):
    """Parse package name and version constraint from string
    Here is a cleaner, more intelligent version of the merge script that:
    deduplicates packages
    prefers the highest / most specific version when multiple are present
    handles the same package with different operators (== > >= > >)
    removes repeated pip entries
    normalizes channels
    ignores -e (editable installs) unless you really need them"""
    dep_str = dep_str.strip()
    if not dep_str or dep_str.startswith('#') or dep_str == '-e ..':
        return None, None, None

    # Match package[op]version
    match = re.match(r'^([a-zA-Z0-9\-_\.]+)\s*([<>=!]+)?\s*([\d\.\*]+.*)?$', dep_str)
    if not match:
        return dep_str, None, None

    name = match.group(1).lower()  # normalize to lowercase for dedup
    op = match.group(2) or ''
    ver_str = match.group(3) or ''
    constraint = f"{op}{ver_str}" if op and ver_str else None
    return name, constraint, dep_str


root = Path(".")
channels = set(["conda-forge", "defaults"])
dep_versions = defaultdict(list)   # pkg_lower → list of (parsed_version, original_string, source)
pip_deps = set()

for yml_path in root.rglob("conda.yml"):
    try:
        content = yaml.safe_load(yml_path.read_text())
    except Exception as e:
        print(f"Skipping {yml_path}: {e}")
        continue

    if not isinstance(content, dict):
        continue

    # Channels
    if "channels" in content:
        channels.update(c.strip() for c in content["channels"] if isinstance(c, str))

    # Dependencies
    if "dependencies" in content:
        for dep in content["dependencies"]:
            if isinstance(dep, str):
                name, constraint, original = parse_dep(dep)
                if name:
                    # Try to parse version for comparison
                    version_obj = None
                    if constraint and constraint.startswith("=="):
                        try:
                            version_obj = Version(constraint[2:].strip())
                        except InvalidVersion:
                            pass
                    dep_versions[name].append((version_obj, original, str(yml_path)))
            elif isinstance(dep, dict) and "pip" in dep:
                for p in dep["pip"]:
                    if isinstance(p, str) and not p.startswith("-e"):
                        pip_deps.add(p.strip())

# Resolve best version per package
final_deps = []

for pkg, entries in dep_versions.items():
    if not entries:
        continue

    # Prefer exact pin (==) with highest version
    exact = [e for e in entries if e[0] is not None]
    if exact:
        exact.sort(key=lambda x: x[0], reverse=True)
        chosen_str = exact[0][1]
        print(f"Selected for {pkg}: {chosen_str} (highest exact version)")
    else:
        # Fall back to first non-empty constraint or just package name
        non_empty = [e for e in entries if e[1]]
        chosen_str = non_empty[0][1] if non_empty else pkg
        print(f"Selected for {pkg}: {chosen_str} (no exact version found)")

    final_deps.append(chosen_str)

# Sort dependencies (cosmetic)
final_deps.sort()

# Build final merged file
merged = {
    "name": "nyc_airbnb_merged",
    "channels": sorted(channels),
    "dependencies": final_deps
}

if pip_deps:
    merged["dependencies"].append({"pip": sorted(pip_deps)})

output_file = root / "merged_conda.yml"
output_file.write_text(yaml.dump(merged, sort_keys=False, allow_unicode=True))

print(f"\nMerged file saved: {output_file}")
print(f"Channels: {', '.join(sorted(channels))}")
print(f"Conda dependencies: {len(final_deps)}")
print(f"Pip dependencies: {len(pip_deps)}")