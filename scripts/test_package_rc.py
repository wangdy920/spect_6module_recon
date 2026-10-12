"""Exercise release identity guards without CUDA, building or creating a package."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def checked(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          encoding="utf-8", **kwargs).stdout.strip()


def main():
    source = Path(__file__).with_name("package_rc.ps1")
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell is required"
    with TemporaryDirectory(prefix="spect_release_guard_") as tmp:
        root = Path(tmp)
        repo = root / "repo"
        scripts = repo / "scripts"
        scripts.mkdir(parents=True)
        target = scripts / source.name
        shutil.copy2(source, target)
        checked(["git", "init", "-q", str(repo)])
        checked(["git", "add", "scripts/package_rc.ps1"], cwd=repo)
        checked(["git", "-c", "user.name=Release Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "fixture"], cwd=repo)
        commit = checked(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
        staging = root / "staging"
        binary = staging / "bin/SPECT_recon.exe"
        binary.parent.mkdir(parents=True)
        # A tiny managed EXE lets the real packaging script inspect controlled versions.
        code = ('public class VersionFixture { public static void Main() { '
                'System.Console.WriteLine(System.Environment.GetEnvironmentVariable("SPECT_TEST_VERSION")); '
                'System.Environment.ExitCode = int.Parse(System.Environment.GetEnvironmentVariable("SPECT_TEST_EXIT")); } }')
        escaped = str(binary).replace("'", "''")
        checked([powershell, "-NoProfile", "-Command",
                 f"Add-Type -TypeDefinition '{code}' -OutputAssembly '{escaped}' -OutputType ConsoleApplication"])
        command = [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(target),
                   "-SkipBuild", "-DryRun", "-StagingRoot", str(staging)]
        cases = [
            (f"20000101+{commit}", "0", True, "clean Git worktree"),
            (f"20000101+{commit}-dirty", "0", False, "does not identify clean commit"),
            ("20000101+not_the_commit", "0", False, "does not identify clean commit"),
            ("", "0", False, "does not identify clean commit"),
            (f"20000101+{commit}", "9", False, "--version failed"),
            (f"20000101+{commit}", "0", False, None),
        ]
        for version, exit_code, dirty, error in cases:
            target.write_bytes(source.read_bytes() + (b"\n# test modification\n" if dirty else b""))
            env = dict(os.environ, SPECT_TEST_VERSION=version, SPECT_TEST_EXIT=exit_code)
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env)
            output = result.stdout + result.stderr
            assert (result.returncode == 0) == (error is None), output
            assert error is None or error in output, output
            assert not (repo / "dist").exists(), "Preflight must not create or delete package output"
        print(f"PASS: {len(cases)} release identity checks")


if __name__ == "__main__":
    main()
