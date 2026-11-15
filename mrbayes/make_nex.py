#!/usr/bin/env python3
"""Regenerate treeN/treeN.nex for the DS1 fixed-topology MrBayes runs.

The 42 launch files share one template (alignment + TREES block + MRBAYES
block) and differ only in the fixed topology. This script reassembles them
from two sources:

  --src/DS1_fixed_tree.nex -- template; the line "__MYSTARTTREE_LINE__"
                              is replaced by the fixed-topology declaration
  --src/DS1_topologies.txt  -- 42 Newick strings, one per line (tree1..tree42)

Usage:
  python3 make_nex.py --root /path/to/workdir            # all 42 trees
  python3 make_nex.py --root /path/to/workdir --tree-ids "1 7 42"
"""
import argparse
from pathlib import Path

PLACEHOLDER = "__MYSTARTTREE_LINE__"
LINE_PREFIX = "  tree mystarttree = [&U] "
DEFAULT_IDS = " ".join(str(i) for i in range(1, 43))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(Path(__file__).resolve().parent),
                   help="directory containing DS1_fixed_tree.nex and "
                        "DS1_topologies.txt (default: this script's directory)")
    p.add_argument("--root", required=True,
                   help="output directory; treeN/treeN.nex are written here")
    p.add_argument("--tree-ids", default=DEFAULT_IDS,
                   help="space-separated tree ids to generate (default: 1..42)")
    args = p.parse_args()

    src = Path(args.src)
    template = (src / "DS1_fixed_tree.nex").read_text()
    if template.count(PLACEHOLDER) != 1:
        raise SystemExit("template must contain exactly one " + PLACEHOLDER)
    topos = (src / "DS1_topologies.txt").read_text().splitlines()
    if len(topos) != 42:
        raise SystemExit("DS1_topologies.txt must have 42 lines, got %d"
                         % len(topos))

    root = Path(args.root)
    for i in (int(t) for t in args.tree_ids.split()):
        topo = topos[i - 1].strip()
        if not (topo.startswith("(") and topo.endswith(")")):
            raise SystemExit("line %d is not a Newick tree: %r" % (i, topo[:60]))
        body = template.replace(PLACEHOLDER, LINE_PREFIX + topo + ";")
        outdir = root / ("tree%d" % i)
        outdir.mkdir(exist_ok=True)
        (outdir / ("tree%d.nex" % i)).write_text(body)
        print("wrote tree%d/tree%d.nex" % (i, i))


if __name__ == "__main__":
    main()
