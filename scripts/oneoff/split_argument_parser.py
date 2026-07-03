#!/usr/bin/env python3
"""Move ArgumentParser inline implementations from header to src/cli/ArgumentParser.cpp."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = ROOT / "include/cli/ArgumentParser.h"
CPP_PATH = ROOT / "src/cli/ArgumentParser.cpp"


def main() -> None:
    text = HEADER_PATH.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    class_end = None
    for i, line in enumerate(lines):
        if line.strip() == "};" and i > 100:
            class_end = i
            break
    if class_end is None:
        raise SystemExit("class end not found")

    impl_start = None
    for i in range(class_end + 1, len(lines)):
        if lines[i].lstrip().startswith("inline "):
            impl_start = i
            break
    if impl_start is None:
        raise SystemExit("impl start not found")

    header_part = "".join(lines[: class_end + 1])
    impl_lines = lines[impl_start:]

    while impl_lines:
        stripped = impl_lines[-1].strip()
        if stripped == "":
            impl_lines.pop()
            continue
        if stripped.startswith("} // namespace"):
            impl_lines.pop()
            continue
        break

    impl_body = "".join(impl_lines).replace("inline ", "")

    cpp = (
        "// ArgumentParser implementation (moved from header, PR15 move-only).\n\n"
        '#include "cli/ArgumentParser.h"\n\n'
        "namespace spect {\n"
        "namespace cli {\n\n"
        f"{impl_body}\n"
        "} // namespace cli\n"
        "} // namespace spect\n"
    )

    header = header_part.rstrip() + "\n\n} // namespace cli\n} // namespace spect\n"

    CPP_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(header, encoding="utf-8", newline="\n")
    CPP_PATH.write_text(cpp, encoding="utf-8", newline="\n")
    print(f"Wrote {HEADER_PATH} ({len(header.splitlines())} lines)")
    print(f"Wrote {CPP_PATH} ({len(cpp.splitlines())} lines)")


if __name__ == "__main__":
    main()
