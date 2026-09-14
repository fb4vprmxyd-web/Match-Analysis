"""CLI: turn a free-text sample description into a pre-match cache selection.

    python3 interpret_selection.py --opp "Bradford City"
    python3 interpret_selection.py --opp "Bradford City" --notes "last 5 cup games for each team"

Interprets via selector_interpreter.interpret_selection(), prints a plain-
English readback for confirmation, then resolves both sides through
match_selector.resolve_selection() (pulling anything not already cached)
and writes prematch_env.sh with the PRE_OPP / PRE_OPP_CACHES / PRE_OUR_CACHES
exports build_prematch.py already reads - build_prematch.py itself is
untouched, this just fills in its env-var contract.

    source prematch_env.sh && python3 build_prematch.py
"""

import argparse
import sys

from envconfig import load_match, DATA_ROOT
from match_selector import resolve_selection
from selector_interpreter import interpret_selection, readback


def _read_notes(args) -> str:
    if args.notes:
        return args.notes
    if args.notes_file:
        return open(args.notes_file).read()
    print("Describe the sample (e.g. \"last 10 games from each team\"), "
         "then Ctrl-D:", file=sys.stderr)
    return sys.stdin.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp", required=True, help="Upcoming opponent's name")
    ap.add_argument("--notes", help="Sample description as a single argument")
    ap.add_argument("--notes-file", help="Path to a text file with the description")
    ap.add_argument("--out", default=str(DATA_ROOT / "prematch_env.sh"),
                   help="Where to write the shell exports (default: %(default)s)")
    args = ap.parse_args()

    cfg = load_match()
    us = cfg.get("team", "Leyton Orient")
    raw_text = _read_notes(args)

    specs = interpret_selection(raw_text, us, args.opp)

    while True:
        print(f"\n--- sample for {us} v {args.opp} ---")
        for line in readback(specs):
            print(" ", line)
        choice = input("\nResolve these against the live fixture list? "
                       "[y]es / [n]o / [e]dit counts: ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("Aborted - nothing resolved.")
            return
        if choice == "e":
            for side in ("opponent", "us"):
                c = input(f"  {side} count [{specs[side]['count']}]: ").strip()
                if c:
                    specs[side]["count"] = int(c)
                comp = input(f"  {side} competition (all/league/cup) "
                            f"[{specs[side]['competition']}]: ").strip()
                if comp:
                    specs[side]["competition"] = comp
            continue
        print("Please answer y, n, or e.")

    opp_caches = resolve_selection(cfg, specs["opponent"])
    our_caches = resolve_selection(cfg, specs["us"])
    if not opp_caches or not our_caches:
        sys.exit("Resolved zero matches for one side - check the opponent "
                 "name and that fixtures are published yet.")

    lines = [
        f'export PRE_OPP="{args.opp}"',
        f'export PRE_OPP_CACHES="{",".join(opp_caches)}"',
        f'export PRE_OUR_CACHES="{",".join(our_caches)}"',
    ]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nWrote {args.out}:")
    for line in lines:
        print(" ", line)
    print(f"\n  source {args.out} && python3 build_prematch.py")


if __name__ == "__main__":
    main()
