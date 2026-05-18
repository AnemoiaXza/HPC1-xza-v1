#!/usr/bin/env python3
"""HPC Experiment 1 - Visualization of matrix multiplication performance."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def plot_speed_comparison(df, outdir):
    """Bar chart: all methods at 2048x2048"""
    sz = 2048
    d = df[df['size'] == sz].copy()
    if d.empty:
        sz = df['size'].max()
        d = df[df['size'] == sz].copy()

    # pick best worker count for each method
    methods = {
        'serial': ('Serial', d[d['method'] == 'serial']),
        'openmp': ('OpenMP', d[d['method'] == 'openmp']),
        'mpi-shm': ('MPI-SharedMem', d[d['method'] == 'mpi-shm']),
        'mpi-original': ('MPI-Original', d[d['method'] == 'mpi-original']),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: best time per method
    labels, times = [], []
    for name, (label, sub) in methods.items():
        if sub.empty:
            continue
        best = sub.loc[sub['time_ms'].idxmin()]
        labels.append(label)
        times.append(best['time_ms'])
        workers = int(best['workers']) if name != 'serial' else 1
        axes[0].bar(label, best['time_ms'],
                    color=['#2c3e50', '#27ae60', '#2980b9', '#e74c3c'][len(labels)-1])
        axes[0].text(labels.index(label), best['time_ms'] + max(times)*0.02,
                     f"{best['time_ms']:.0f}ms\n({workers}w)",
                     ha='center', fontsize=9)

    axes[0].set_title(f'Best Time per Method ({sz}x{sz})', fontsize=13)
    axes[0].set_ylabel('Time (ms)')
    axes[0].grid(axis='y', alpha=0.3)

    # Right: speedup relative to serial
    serial_time = d[d['method'] == 'serial']['time_ms'].values[0]
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Serial baseline')

    colors = {'openmp': '#27ae60', 'mpi-shm': '#2980b9', 'mpi-original': '#e74c3c'}
    markers = {'openmp': 'o', 'mpi-shm': 's', 'mpi-original': '^'}

    for method in ['openmp', 'mpi-shm', 'mpi-original']:
        sub = d[d['method'] == method]
        if sub.empty:
            continue
        sub = sub.sort_values('workers')
        speedup = serial_time / sub['time_ms'].values
        axes[1].plot(sub['workers'].values, speedup, marker=markers[method],
                     color=colors[method], linewidth=2, markersize=8, label=method)

    axes[1].set_title(f'Speedup vs Workers ({sz}x{sz})', fontsize=13)
    axes[1].set_xlabel('Number of Workers (threads/processes)')
    axes[1].set_ylabel('Speedup (vs Serial)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'speed_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def plot_scaling(df, outdir):
    """Line chart: time vs matrix size for each method"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # use best worker config for each method at each size
    colors = {'serial': '#2c3e50', 'openmp': '#27ae60',
              'mpi-shm': '#2980b9', 'mpi-original': '#e74c3c'}
    markers = {'serial': 'D', 'openmp': 'o', 'mpi-shm': 's', 'mpi-original': '^'}

    for method in ['serial', 'openmp', 'mpi-shm', 'mpi-original']:
        sub = df[df['method'] == method]
        if sub.empty:
            continue
        best_times = sub.loc[sub.groupby('size')['time_ms'].idxmin()]
        best_times = best_times.sort_values('size')
        ax.plot(best_times['size'].values, best_times['time_ms'].values,
                marker=markers[method], color=colors[method],
                linewidth=2, markersize=8, label=method)

    ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Scaling with Matrix Size (best worker count per method)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(outdir, 'scaling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def plot_efficiency(df, outdir):
    """Efficiency analysis: where does parallelism help/hurt"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 512 size (small - comm overhead dominates)
    # Right: 4096 size (large - compute dominates)
    for idx, sz in enumerate([512, 4096]):
        ax = axes[idx]
        d = df[df['size'] == sz]
        if d.empty:
            continue
        serial_t = d[d['method'] == 'serial']['time_ms'].values[0]

        methods_data = {
            'OpenMP': d[d['method'] == 'openmp'].sort_values('workers'),
            'MPI-SharedMem': d[d['method'] == 'mpi-shm'].sort_values('workers'),
        }

        for label, sub in methods_data.items():
            if sub.empty:
                continue
            speedup = serial_t / sub['time_ms'].values
            color = '#27ae60' if 'OpenMP' in label else '#2980b9'
            ax.plot(sub['workers'].values, speedup, 'o-', color=color,
                    linewidth=2, markersize=8, label=label)

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Serial')
        ax.fill_between([0, 17], 0, 1, alpha=0.05, color='red')
        ax.fill_between([0, 17], 1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5,
                        alpha=0.05, color='green')
        ax.set_title(f'Size {sz}x{sz}', fontsize=13)
        ax.set_xlabel('Workers')
        ax.set_ylabel('Speedup')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'efficiency.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'bench_result.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark.sh first.", file=sys.stderr)
        sys.exit(1)

    df = load_data(csv_path)
    outdir = script_dir
    plot_speed_comparison(df, outdir)
    plot_scaling(df, outdir)
    plot_efficiency(df, outdir)
    print("All plots generated.")

if __name__ == '__main__':
    main()
