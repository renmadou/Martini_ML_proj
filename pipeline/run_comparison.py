"""Cross-tool comparison driver: Martini Mapper vs MolCluster on a set of molecules.

Run in any env with conda available (base is fine). Bridges the two conda envs:

  1. martini_mapper env: maps each SMILES to Martini beads (default xtb flow,
     or --no-xtb), parses the bead count and atom assignment from <Name>.txt
  2. molcluster env: Leiden bisection search with N_target = Martini bead count
  3. writes a markdown comparison table to benchmarks/<tag>_comparison.md

Usage:
    python pipeline/run_comparison.py molecules.txt [--tag week2] [--no-xtb]

molecules.txt format (one per line, '#' comments allowed):
    Benzene c1ccccc1
"""
import argparse
import json
import os
import re
import subprocess
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def martini_beads(name, smiles, no_xtb, out_dir):
    cmd = ['conda', 'run', '-n', 'martini_mapper', 'python', '-m', 'martini_mapper',
           name, smiles, '--out-dir', out_dir]
    if no_xtb:
        cmd.append('--no-xtb')
    r = run(cmd)
    if r.returncode != 0:
        return {'error': (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else 'unknown'}
    txt = os.path.join(out_dir, f'{name}.txt')
    beads = []
    with open(txt) as f:
        in_table = False
        for line in f:
            if line.startswith('index') and 'bead_type' in line:
                in_table = True
                continue
            if in_table:
                if not line.strip() or line.startswith('Bead connections'):
                    break
                parts = line.split()
                beads.append({'idx': int(parts[0]), 'bead_type': parts[1],
                              'atoms': [int(a) for a in parts[2].split(',')]})
    return {'n_beads': len(beads), 'beads': beads}


def molcluster_groups(smiles, n_target, tol=1):
    r = run(['conda', 'run', '-n', 'molcluster', 'python',
             os.path.join(PROJ_ROOT, 'pipeline', 'molcluster_bisect.py'),
             '--smiles', smiles, '--n-target', str(n_target), '--tol', str(tol)])
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return {'converged': False, 'error': (r.stderr or r.stdout).strip().splitlines()[-1]
            if (r.stderr or r.stdout) else 'no output'}


def fmt_assignment(beads):
    return ' '.join('{' + ','.join(str(a) for a in b['atoms']) + '}' for b in beads)


def fmt_groups(groups, n_beads):
    if groups is None:
        return '-'
    clusters = {}
    for atom, g in enumerate(groups):
        clusters.setdefault(g, []).append(atom)
    return ' '.join('{' + ','.join(str(a) for a in sorted(atoms)) + '}'
                    for _, atoms in sorted(clusters.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('molecules', help='text file: "<Name> <SMILES>" per line')
    p.add_argument('--tag', default='comparison')
    p.add_argument('--tol', type=int, default=1)
    p.add_argument('--no-xtb', action='store_true')
    args = p.parse_args()

    mols = []
    with open(args.molecules) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                name, smiles = line.split(None, 1)
                mols.append((name, smiles.strip()))

    out_dir = os.path.join(PROJ_ROOT, 'benchmarks', 'mapper_outputs')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, smiles in mols:
        print(f'[{name}] Martini Mapper ...', flush=True)
        m = martini_beads(name, smiles, args.no_xtb, out_dir)
        if 'error' in m:
            rows.append((name, smiles, None, None, f"Mapper failed: {m['error']}"))
            continue
        print(f'[{name}] {m["n_beads"]} beads -> MolCluster bisection ...', flush=True)
        g = molcluster_groups(smiles, m['n_beads'], tol=args.tol)
        note = 'converged' if g.get('converged') else f"NOT converged: {g.get('error', '?')}"
        rows.append((name, smiles, m, g, note))

    md = os.path.join(PROJ_ROOT, 'benchmarks', f'{args.tag}_comparison.md')
    with open(md, 'w') as f:
        f.write(f'# Martini Mapper vs MolCluster ({args.tag})\n\n')
        f.write('| Molecule | SMILES | Mapper beads | MolCluster N_groups | bisect iters | converged gamma | Mapper assignment | MolCluster assignment | note |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        for name, smiles, m, g, note in rows:
            if m is None:
                f.write(f'| {name} | `{smiles}` | - | - | - | - | - | - | {note} |\n')
                continue
            f.write(f"| {name} | `{smiles}` | {m['n_beads']} | {g.get('n_groups', '-')} | "
                    f"{g.get('iterations', '-')} | {g.get('gamma') or '-'} | "
                    f"{fmt_assignment(m['beads'])} | {fmt_groups(g.get('groups'), m['n_beads'])} | {note} |\n")
    print(f'\nreport -> {md}')


if __name__ == '__main__':
    main()
