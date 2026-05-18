#!/usr/bin/env python3
"""Visualization for three MPI experiments: gemm, conv, pooling."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(csv_path):
    return pd.read_csv(csv_path)

def plot_scaling_all(df, outdir):
    """Scaling comparison: speedup vs np for all experiments at 1024x1024"""
    sz = 1024
    d = df[df['size'] == sz].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute time
    experiments = ['gemm', 'conv-naive', 'conv-img2col', 'pooling']
    labels = ['Gemm (matmul)', 'Conv (naive)', 'Conv (img2col)', 'Pooling']
    colors = ['#2c3e50', '#e74c3c', '#27ae60', '#2980b9']
    markers = ['o', '^', 's', 'D']

    for exp, label, color, marker in zip(experiments, labels, colors, markers):
        sub = d[d['experiment'] == exp].sort_values('workers')
        if sub.empty or len(sub) < 2:
            continue
        axes[0].plot(sub['workers'].values, sub['time_ms'].values,
                     marker=marker, color=color, linewidth=2, markersize=8,
                     label=label)
        # annotate values
        for _, row in sub.iterrows():
            axes[0].annotate(f"{row['time_ms']:.1f}",
                            (row['workers'], row['time_ms']),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=7, ha='center', color=color)

    axes[0].set_title(f'Absolute Time vs Process Count ({sz}x{sz})', fontsize=13)
    axes[0].set_xlabel('Number of MPI Processes')
    axes[0].set_ylabel('Time (ms)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right: speedup relative to np=1
    for exp, label, color, marker in zip(experiments, labels, colors, markers):
        sub = d[d['experiment'] == exp].sort_values('workers')
        if sub.empty or len(sub) < 2:
            continue
        baseline = sub[sub['workers'] == sub['workers'].min()]['time_ms'].values[0]
        if baseline == 0:
            continue
        speedup = baseline / sub['time_ms'].values
        axes[1].plot(sub['workers'].values, speedup,
                     marker=marker, color=color, linewidth=2, markersize=8,
                     label=label)

    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='np=1 baseline')
    axes[1].fill_between([0, 9], 0, 1, alpha=0.05, color='red')
    axes[1].fill_between([0, 9], 1, axes[1].get_ylim()[1], alpha=0.05, color='green')
    axes[1].set_title(f'Speedup vs np=1 ({sz}x{sz})', fontsize=13)
    axes[1].set_xlabel('Number of MPI Processes')
    axes[1].set_ylabel('Speedup (relative to np=1)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, '3exp_scaling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def plot_conv_comparison(df, outdir):
    """Conv: naive vs img2col across sizes and process counts"""
    conv = df[df['experiment'].str.contains('conv')].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = sorted(conv['size'].unique())
    colors_map = {'conv-naive': '#e74c3c', 'conv-img2col': '#27ae60'}

    # Left: time vs size for np=1
    ax = axes[0]
    for exp in ['conv-naive', 'conv-img2col']:
        sub = conv[(conv['experiment'] == exp) & (conv['workers'] == 1)].sort_values('size')
        if sub.empty:
            continue
        label = 'Naive Conv' if 'naive' in exp else 'img2col Conv'
        ax.plot(sub['size'].values, sub['time_ms'].values, 'o-',
                color=colors_map[exp], linewidth=2, markersize=8, label=label)

    ax.set_title('Conv: Naive vs img2col (np=1)', fontsize=13)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: speedup from img2col at each size (np=1)
    ax = axes[1]
    for size in sizes:
        nv = conv[(conv['experiment'] == 'conv-naive') & (conv['workers'] == 1) & (conv['size'] == size)]
        im = conv[(conv['experiment'] == 'conv-img2col') & (conv['workers'] == 1) & (conv['size'] == size)]
        if len(nv) and len(im):
            sp = nv['time_ms'].values[0] / im['time_ms'].values[0]
            ax.bar(str(size), sp, color='#27ae60', alpha=0.8)
            ax.text(str(size), sp + 0.1, f'{sp:.1f}x', ha='center', fontweight='bold')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('img2col Speedup over Naive (np=1)', fontsize=13)
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Matrix Size')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'conv_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def plot_comm_compute_ratio(df, outdir):
    """Communication-computation ratio analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: compute intensity comparison
    experiments = ['gemm', 'conv-naive', 'pooling']
    labels = ['Gemm O(n³)', 'Conv O(n²·k²)', 'Pooling O(n²)']
    colors = ['#2c3e50', '#e74c3c', '#2980b9']
    sizes = [512, 1024, 2048]

    ax = axes[0]
    x = range(len(sizes))
    width = 0.25

    for i, (exp, label, color) in enumerate(zip(experiments, labels, colors)):
        times = []
        for sz in sizes:
            sub = df[(df['experiment'] == exp) & (df['size'] == sz) & (df['workers'] == 1)]
            times.append(sub['time_ms'].values[0] if len(sub) else 0)
        # normalize to 512
        if times[0] > 0:
            normalized = [t / times[0] for t in times]
            ax.bar([xi + i*width for xi in x], normalized, width, label=label, color=color, alpha=0.8)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_title('Time Growth: 512→1024→2048 (normalized to 512)', fontsize=12)
    ax.set_ylabel('Normalized Time')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # annotate theoretical ratios
    ax.axhline(y=8, color='gray', linestyle=':', alpha=0.5)
    ax.text(2.5, 8, 'O(n³) baseline=8x', fontsize=8, color='gray')

    # Right: communication vs computation per experiment
    ax = axes[1]
    # metric: (time_np4 - time_np1) / time_np1  * 100 => comm overhead percentage
    for exp, label, color in zip(experiments, labels, colors):
        overheads = []
        sizes_used = []
        for sz in sizes:
            t1 = df[(df['experiment'] == exp) & (df['size'] == sz) & (df['workers'] == 1)]
            t4 = df[(df['experiment'] == exp) & (df['size'] == sz) & (df['workers'] == 4)]
            if len(t1) and len(t4) and t1['time_ms'].values[0] > 0:
                overhead = (t4['time_ms'].values[0] - t1['time_ms'].values[0]) / t1['time_ms'].values[0] * 100
                overheads.append(overhead)
                sizes_used.append(sz)

        if overheads:
            ax.plot(sizes_used, overheads, 'o-', color=color, linewidth=2, markersize=8, label=label)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Communication Overhead (np=4 vs np=1)', fontsize=13)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Overhead (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'comm_compute_ratio.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'bench_3exp.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark_3exp.sh first.")
        return

    df = load_data(csv_path)
    outdir = script_dir

    plot_scaling_all(df, outdir)
    plot_conv_comparison(df, outdir)
    plot_comm_compute_ratio(df, outdir)
    print("All 3exp plots generated.")

if __name__ == '__main__':
    main()
