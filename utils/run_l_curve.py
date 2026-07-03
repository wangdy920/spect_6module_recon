
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def parse_metrics(output):
    # Look for [METRIC] Beta=... Likelihood=... Penalty=... Objective=...
    match = re.search(r'\[METRIC\] Beta=([\d\.e\-\+]+) Likelihood=([\d\.e\-\+]+) Penalty=([\d\.e\-\+]+)', output)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None

def load_args_file(path: str):
    args = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        args.append(s)
    return args

def main():
    parser = argparse.ArgumentParser(description='Run L-Curve Analysis for SPECT Reconstruction')
    parser.add_argument('--exe', required=True, help='Path to SPECT_recon executable')
    parser.add_argument('--base_args', default='', help='Base arguments (excluding -reg). NOTE: simple whitespace split; prefer --base_arg for paths with spaces.')
    parser.add_argument('--base_arg', action='append', default=[], help='Repeatable base argument token. Recommended for Windows paths with spaces. Example: --base_arg -df --base_arg \"D:/data/file.mat\"')
    parser.add_argument('--base_args_file', default='', help='Text file containing one argument token per line (empty lines and # comments ignored).')
    parser.add_argument('--reg_type', default='TVIsotropic', help='Regularization type (e.g. TVIsotropic, NLM)')
    parser.add_argument('--betas', required=True, help='Comma separated list of betas or start:end:steps (logspace)')
    parser.add_argument('--reg_params', default='', help='Additional reg params (e.g. tv_eps or nlm_h, etc)')
    parser.add_argument('--output', default='l_curve_results.png', help='Output plot filename')
    
    args = parser.parse_args()

    # Resolve base args tokens (exclude -reg)
    base_args_tokens = []
    if args.base_args_file:
        base_args_tokens = load_args_file(args.base_args_file)
    elif args.base_arg:
        base_args_tokens = args.base_arg
    elif args.base_args:
        base_args_tokens = args.base_args.split()
    else:
        print("Error: one of --base_arg/--base_args_file/--base_args must be provided.", file=sys.stderr)
        sys.exit(2)

    # Parse betas
    beta_list = []
    if ':' in args.betas:
        parts = args.betas.split(':')
        start, end, steps = float(parts[0]), float(parts[1]), int(parts[2])
        beta_list = np.logspace(np.log10(start), np.log10(end), steps).tolist()
    else:
        beta_list = [float(x) for x in args.betas.split(',')]

    results = []
    
    print(f"Running L-Curve analysis for {len(beta_list)} beta values: {beta_list}")

    for beta in beta_list:
        # Construct reg argument
        # Format: -reg Type,Beta,Param1,Param2...
        reg_arg_str = f"{args.reg_type},{beta}"
        if args.reg_params:
            reg_arg_str += f",{args.reg_params}"
        
        cmd = [args.exe] + base_args_tokens + ["-reg", reg_arg_str]
        
        print(f"Running beta={beta:.4e} ...")
        try:
            # Capture stdout
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            metrics = parse_metrics(output)
            if metrics:
                b, lik, pen = metrics
                results.append({'beta': b, 'likelihood': lik, 'penalty': pen})
                print(f"  -> Likelihood={lik:.2e}, Penalty={pen:.2e}")
            else:
                print("  -> Failed to parse metrics from output.")
        except subprocess.CalledProcessError as e:
            print(f"  -> Execution failed: {e}")
            print(e.stdout)
            print(e.stderr)

    if not results:
        print("No successful runs.")
        return

    # Plot
    betas = [r['beta'] for r in results]
    liks = [r['likelihood'] for r in results]
    pens = [r['penalty'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(pens, liks, 'o-')
    
    # Annotate betas
    for i, txt in enumerate(betas):
        plt.annotate(f'{txt:.1e}', (pens[i], liks[i]), fontsize=8)

    plt.title(f'L-Curve Analysis ({args.reg_type})')
    plt.xlabel('Regularization Penalty R(f)')
    plt.ylabel('Log-Likelihood L(f)')
    plt.grid(True)
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

    # Suggest "corner" (simple curvature max or simple visualization)
    print("\nResults Table:")
    print(f"{'Beta':<12} {'Likelihood':<15} {'Penalty':<15}")
    for r in results:
        print(f"{r['beta']:<12.4e} {r['likelihood']:<15.4e} {r['penalty']:<15.4e}")

if __name__ == "__main__":
    main()
