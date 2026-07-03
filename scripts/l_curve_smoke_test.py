import tempfile
from pathlib import Path
import importlib.util


def load_run_l_curve_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "utils" / "run_l_curve.py"
    spec = importlib.util.spec_from_file_location("run_l_curve", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_run_l_curve_module()

    sample = "[METRIC] Beta=1e-3 Likelihood=-12.5 Penalty=3.25 Objective=-15.75 Model=runtime"
    metrics = mod.parse_metrics(sample)
    if metrics is None:
        raise AssertionError("parse_metrics failed to match sample output")
    beta, likelihood, penalty = metrics
    if abs(beta - 1e-3) > 1e-12 or abs(likelihood + 12.5) > 1e-12 or abs(penalty - 3.25) > 1e-12:
        raise AssertionError(f"parse_metrics values mismatch: {metrics}")

    with tempfile.TemporaryDirectory() as td:
        args_path = Path(td) / "args.txt"
        args_path.write_text(
            "\n".join(
                [
                    "# comment",
                    "",
                    "-df",
                    "data.mat",
                    "-it",
                    "2:4",
                ]
            ),
            encoding="utf-8",
        )
        tokens = mod.load_args_file(str(args_path))
        if tokens != ["-df", "data.mat", "-it", "2:4"]:
            raise AssertionError(f"load_args_file tokens mismatch: {tokens}")

    print("L-curve smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
