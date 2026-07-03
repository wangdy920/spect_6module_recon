#!/usr/bin/env python3
"""Split tests/test_interfaces.cpp into domain-specific files (PR24)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tests" / "test_interfaces.cpp"

MATRIX_IO = {
    "ProjectionDataEmpty",
    "ReorderProjection3D",
    "ReorderProjectionLimitAngles",
    "LoadProjectionBinAutoDetects",
    "IoCSRBv1HeaderIs24Bytes",
    "ReorderProjectionEmptyThrows",
}

COMPACT_SYMMETRY = {
    "CompactSymmetryMetadataReadsLegacyLayout",
    "CompactSymmetryMapsSixModuleFullProjectionRowsOnce",
    "CompactSymmetryValidationRejectsCompactProjectionRows",
    "CompactSymmetryMetadataMissingFileThrows",
    "CompactSymmetryMetadataRejectsTruncatedExtendedHeader",
    "CompactSymmetryMetadataRejectsUnexpectedTrailingBytes",
    "Real260601MetadataParsesWhenPresent",
    "Real260601Sym90ZCoversAllFullProjectionRowsWhenPresent",
    "Real260601ChunkDirectoryReadsWhenPresent",
    "CompactSymmetryMapsSixModule90ZFullProjectionRowsOnce",
    "CompactChunkCSRDirectoryReadsToyChunks",
    "CompactChunkCSRSampleValidationRejectsTailOutOfRangeColIdx",
    "CompactSymmetryToyMetadataAndD4MappingMatchLegacyRules",
    "CompactSymmetryToyExpansionCoversFullProjectionOnce",
    "CompactSymmetryToyNumeratorDenominatorMatchExplicitFull",
    "CompactSymmetryToyMlemOneIterationMatchesExplicitFull",
    "CompactSymmetryToyZeroPsfIsIdentityAndGaussianCommutesWithD4",
}

MULTIBED = {
    "MultiBedParserSortsDfListAndBProjListByBedNumber",
    "MultiBedParserRejectsMoreThanFiveDfListBeds",
    "MultiBedParserRejectsMoreThanFiveBedOverlapSlots",
    "MultiBedParserParsesZFlipSwitches",
    "MultiBedConfigReadsOverlapCropAndZFlipDefaults",
}

PARSER_IO = {
    "CompactSymmetricParserDefaultsToStandard",
    "StandardParserDoesNotEnableCompactFromMetaOverride",
    "CompactSymmetricParserAcceptsExplicitMetaOverride",
    "CompactSymmetricParserAcceptsDenomCachePath",
    "CompactSymmetricParserAcceptsChunkCSRDirWithoutManualGeometry",
    "CompactSymmetricParserReadsDefaultChunkCSRDirFromConfig",
    "ExplicitLoadCSROverridesConfigDefaultChunkCSRDir",
    "PsfFlagDefaultsToOperatorMode",
    "ExplicitPsfModeNoneOverridesPsfDefaultOperator",
    "CompactSymmetricParserRejectsLoadCSRAndChunkCSRDirTogether",
    "StandardParserRejectsCompactDenomCachePathWithoutCompactLayout",
    "CompactSymmetricParserRejectsUnsupportedPhase1Options",
    "CompactSymmetricParserAcceptsBSREMRDP",
    "CompactSymmetricParserAcceptsBackgroundProjection",
    "ConfigParserStripsInlineComments",
}

FILE_HEADER = '''/**
 * @file {filename}
 * @brief Split from test_interfaces.cpp ({domain}).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

{includes}

{namespace_open}
'''

FILE_FOOTER = '''
{namespace_close}
'''


def namespace_open(*parts: str) -> str:
    return "".join(f"namespace {part} {{\n" for part in parts)


def namespace_close(*parts: str) -> str:
    return "".join(f"}} // namespace {part}\n" for part in reversed(parts))


def categorize(suite: str, name: str) -> str:
    if suite == "CLITest":
        return "parser"
    if suite == "IOTest":
        if name in MATRIX_IO:
            return "matrix_io"
        if name in COMPACT_SYMMETRY:
            return "compact_symmetry"
        if name in MULTIBED:
            return "multibed"
        if name in PARSER_IO:
            return "parser"
        raise ValueError(f"Uncategorized IOTest: {name}")
    return "core"


def remove_anonymous_namespace_block(lines: list[str]) -> list[str]:
    """Drop helper-only anonymous namespace; toy tests inside it are kept."""
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "namespace {":
            start = i
            break
    if start is None:
        return lines

    first_test_inside = None
    for i in range(start + 1, len(lines)):
        if re.match(r"TEST\(", lines[i]):
            first_test_inside = i
            break

    end = start
    depth = 0
    for j in range(start, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        end = j
        if j > start and depth == 0:
            break

    if first_test_inside is None:
        return lines[:start] + lines[end + 1 :]

    return lines[:start] + lines[first_test_inside:end] + lines[end + 1 :]


def find_test_starts(lines: list[str]) -> list[tuple[int, str, str]]:
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"TEST\((\w+),\s*(\w+)\)", line)
        if m:
            out.append((i, m.group(1), m.group(2)))
    return out


def extract_test_bodies(lines: list[str]) -> list[tuple[str, str, str, str]]:
    starts = find_test_starts(lines)
    bodies = []
    for idx, (start, suite, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        chunk = lines[start:end]
        depth = 0
        started = False
        test_end = len(chunk)
        for i, line in enumerate(chunk):
            if "{" in line:
                depth += line.count("{")
                started = True
            if "}" in line:
                depth -= line.count("}")
            if started and depth <= 0:
                test_end = i + 1
                break
        bodies.append((suite, name, categorize(suite, name), "".join(chunk[:test_end])))
    return bodies


def extract_anonymous_helpers(lines: list[str]) -> tuple[str, str]:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "namespace {":
            start = i
            break
    if start is None:
        return "", ""
    depth = 0
    end = start
    for j in range(start, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        end = j
        if j > start and depth == 0:
            break
    block = "".join(lines[start + 1 : end]).strip()
    parser_marker = "bool parseReconArgsForTest"
    pos = block.find(parser_marker)
    compact_body = block[:pos].strip()
    parser_tail = block[pos:].strip()
    test_pos = parser_tail.find("TEST(")
    parser_body = parser_tail[:test_pos].strip() if test_pos >= 0 else parser_tail
    return compact_body, parser_body


def write_helper_headers(compact_body: str, parser_body: str) -> None:
    common = ROOT / "tests" / "common"
    common.mkdir(parents=True, exist_ok=True)
    (common / "compact_symmetry_test_helpers.h").write_text(
        f'''#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "io/CompactSymmetryMetadata.h"

namespace spect {{
namespace test {{

{compact_body}

}} // namespace test
}} // namespace spect
''',
        encoding="utf-8",
    )
    (common / "parser_test_helpers.h").write_text(
        f'''#pragma once

#include <string>
#include <vector>

#include "cli/ArgumentParser.h"

namespace spect {{
namespace test {{

{parser_body}

}} // namespace test
}} // namespace spect
''',
        encoding="utf-8",
    )


def write_split_file(
    filename: str,
    domain: str,
    includes: str,
    bodies: list[str],
    namespaces: tuple[str, ...] = ("spect", "test"),
) -> None:
    text = FILE_HEADER.format(
        filename=filename,
        domain=domain,
        includes=includes,
        namespace_open=namespace_open(*namespaces),
    )
    text += "".join(bodies)
    text += FILE_FOOTER.format(namespace_close=namespace_close(*namespaces))
    (ROOT / "tests" / filename).write_text(text, encoding="utf-8")


def find_section_start(lines: list[str], marker: str) -> int:
    for i, line in enumerate(lines):
        if marker in line:
            return i
    raise ValueError(f"Section marker not found: {marker}")


def build_core_file(lines: list[str]) -> str:
    cli_start = find_section_start(lines, "CLI模块测试")
    app_start = find_section_start(lines, "App模块测试")
    core_lines = lines[:cli_start] + lines[app_start:]
    return "".join(core_lines)


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
    compact_body, parser_body = extract_anonymous_helpers(lines)
    write_helper_headers(compact_body, parser_body)

    sanitized = remove_anonymous_namespace_block(lines)
    bodies = extract_test_bodies(sanitized)
    buckets: dict[str, list[str]] = {
        "core": [],
        "matrix_io": [],
        "compact_symmetry": [],
        "parser": [],
        "multibed": [],
    }
    for _suite, _name, cat, body in bodies:
        buckets[cat].append(body)

    write_split_file(
        "test_matrix_io.cpp",
        "matrix IO",
        '#include "core/CSRHeader.h"\n'
        '#include "io/CSRTypes.h"\n'
        '#include "io/ProjectionLoader.h"\n'
        '#include "io/MatFileReader.h"\n',
        buckets["matrix_io"],
        ("spect", "io", "test"),
    )
    write_split_file(
        "test_compact_symmetry.cpp",
        "compact symmetry",
        '#include "filters/Convolution.h"\n'
        '#include "io/CompactChunkCSR.h"\n'
        '#include "io/CompactSymmetryMetadata.h"\n'
        '#include "tests/common/compact_symmetry_test_helpers.h"\n',
        buckets["compact_symmetry"],
        ("spect", "io", "test"),
    )
    cli_bodies = [b for b in buckets["parser"] if b.startswith("TEST(CLITest,")]
    io_parser_bodies = [b for b in buckets["parser"] if b.startswith("TEST(IOTest,")]
    write_split_file(
        "test_parser_cli.tmp.cpp",
        "CLI",
        '#include "cli/StringUtils.h"\n#include "cli/ConfigParser.h"\n',
        cli_bodies,
        ("spect", "cli", "test"),
    )
    write_split_file(
        "test_parser_io.tmp.cpp",
        "parser IO",
        '#include "cli/ArgumentParser.h"\n'
        '#include "cli/ConfigParser.h"\n'
        '#include "tests/common/parser_test_helpers.h"\n',
        io_parser_bodies,
        ("spect", "io", "test"),
    )
    parser_text = (
        '/**\n * @file test_parser.cpp\n * @brief Split from test_interfaces.cpp (CLI/parser).\n */\n\n'
        + (ROOT / "tests" / "test_parser_cli.tmp.cpp").read_text(encoding="utf-8").split("*/", 1)[1].lstrip()
    )
    parser_text += "\n" + (
        ROOT / "tests" / "test_parser_io.tmp.cpp"
    ).read_text(encoding="utf-8").split("*/", 1)[1].lstrip()
    (ROOT / "tests" / "test_parser.cpp").write_text(parser_text, encoding="utf-8")
    (ROOT / "tests" / "test_parser_cli.tmp.cpp").unlink(missing_ok=True)
    (ROOT / "tests" / "test_parser_io.tmp.cpp").unlink(missing_ok=True)
    write_split_file(
        "test_multibed_cli.cpp",
        "multi-bed CLI",
        '#include "cli/ArgumentParser.h"\n'
        '#include "cli/ConfigParser.h"\n'
        '#include "tests/common/parser_test_helpers.h"\n',
        buckets["multibed"],
        ("spect", "io", "test"),
    )

    SRC.write_text(build_core_file(lines), encoding="utf-8")

    print(f"Tests accounted: {len(bodies)} (was {len(find_test_starts(sanitized))})")
    for key, items in buckets.items():
        print(f"  {key}: {len(items)}")


if __name__ == "__main__":
    main()
