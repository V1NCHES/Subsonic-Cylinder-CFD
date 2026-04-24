import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_convergence_from_data():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Data extracted from user logs
    # Format: {Mach: {Method: (iters, values)}}
    
    # Fast SIP (alpha=0.92) - Outer iterations
    fast_sip = {
        0.1: ([10, 20, 30, 40, 50, 60, 70, 80, 90], [2.889e-02, 7.246e-03, 1.862e-03, 4.926e-04, 1.371e-04, 4.138e-05, 1.393e-05, 5.290e-06, 2.230e-06]),
        0.2: ([10, 20, 30, 40, 50, 60, 70, 80, 90], [2.956e-02, 7.765e-03, 2.093e-03, 5.779e-04, 1.662e-04, 5.102e-05, 1.712e-05, 6.376e-06, 2.623e-06]),
        0.3: ([10, 20, 30, 40, 50, 60, 70, 80, 90], [3.068e-02, 8.718e-03, 2.550e-03, 7.606e-04, 2.332e-04, 7.459e-05, 2.531e-05, 9.248e-06, 3.664e-06]),
        0.4: ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [3.228e-02, 1.027e-02, 3.401e-03, 1.146e-03, 3.927e-04, 1.373e-04, 4.933e-05, 1.835e-05, 7.123e-06, 2.899e-06])
    }

    # SLOR (omega=1.7)
    slor = {
        0.1: ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], [4.594e-03, 1.755e-03, 7.211e-04, 3.089e-04, 1.387e-04, 6.566e-05, 3.282e-05, 1.724e-05, 9.447e-06, 5.347e-06, 3.098e-06, 1.824e-06, 1.086e-06]),
        0.2: ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], [4.667e-03, 1.830e-03, 7.710e-04, 3.372e-04, 1.536e-04, 7.324e-05, 3.664e-05, 1.918e-05, 1.045e-05, 5.884e-06, 3.394e-06, 1.993e-06, 1.185e-06]),
        0.3: ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], [4.790e-03, 1.964e-03, 8.642e-04, 3.925e-04, 1.840e-04, 8.927e-05, 4.494e-05, 2.348e-05, 1.270e-05, 7.077e-06, 4.044e-06, 2.357e-06, 1.394e-06]),
        0.4: ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400], [4.967e-03, 2.171e-03, 1.021e-03, 4.934e-04, 2.435e-04, 1.228e-04, 6.329e-05, 3.339e-05, 1.803e-05, 9.952e-06, 5.608e-06, 3.219e-06, 1.876e-06, 1.108e-06])
    }

    # Parameter Study M=0.3 (up to 5000 iters)
    study_m03 = {
        'Std SIP alpha=0.92': ([50, 100, 500, 1000, 1400, 1412], [4.925e-03, 2.929e-03, 2.126e-04, 9.950e-06, 1.063e-06, 9.982e-07]),
        'Std SIP alpha=0.00': ([50, 500, 1000, 2000, 3000, 4000, 4900, 4944], [2.693e-03, 5.878e-04, 2.683e-04, 6.241e-05, 1.484e-05, 3.627e-06, 1.060e-06, 9.992e-07]),
        'SLOR omega=1.00': ([100, 500, 1000, 2000, 3000, 4000, 4400, 4429], [1.956e-03, 6.107e-04, 2.518e-04, 4.553e-05, 8.977e-06, 1.891e-06, 1.042e-06, 9.988e-07])
    }

    # Iteration counts for legends
    iters = {
        'FastSIP': {0.1: 92, 0.2: 94, 0.3: 98, 0.4: 105},
        'SLOR': {0.1: 1316, 0.2: 1333, 0.3: 1365, 0.4: 1420}
    }
    study_iters = {
        'Std SIP alpha=0.92': 1412,
        'Std SIP alpha=0.00': 4944,
        'SLOR omega=1.00': 4429,
        'Fast SIP alpha=0.92': 98,
        'SLOR omega=1.70': 1365
    }

    # Plot 1: Summary Convergence (Mach Sweep)
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm']
    for i, m in enumerate([0.1, 0.2, 0.3, 0.4]):
        plt.semilogy(fast_sip[m][0], fast_sip[m][1], color=colors[i], marker='o', linestyle='-', label=f'Fast SIP M={m} ({iters["FastSIP"][m]} iters)')
        plt.semilogy(slor[m][0], slor[m][1], color=colors[i], marker='x', linestyle='--', label=f'SLOR M={m} ({iters["SLOR"][m]} iters)')
    
    plt.xlabel('Iterations')
    plt.ylabel('Max Correction |delta_phi|')
    plt.title('Convergence History: Fast SIP vs SLOR')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(ncol=2, fontsize='x-small')
    plt.xlim(0, 1500)
    plt.savefig(results_dir / "convergence_all_updated.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Detailed Parameter Study (M=0.3) up to 5000 iters
    plt.figure(figsize=(10, 6))
    plt.semilogy(fast_sip[0.3][0], fast_sip[0.3][1], 'k-o', linewidth=2, label=f'Fast SIP alpha=0.92 ({study_iters["Fast SIP alpha=0.92"]} iters)')
    plt.semilogy(slor[0.3][0], slor[0.3][1], 'r--x', linewidth=1.5, label=f'SLOR omega=1.70 ({study_iters["SLOR omega=1.70"]} iters)')
    
    for label, (iters_data, vals) in study_m03.items():
        plt.semilogy(iters_data, vals, marker='.', linestyle=':', label=f'{label} ({study_iters[label]} iters)')
    
    plt.xlabel('Iterations')
    plt.ylabel('Max Correction |delta_phi|')
    plt.title('Convergence Study (M=0.3): Influence of Parameters (up to 5000 iters)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize='small')
    plt.xlim(0, 5000)
    plt.savefig(results_dir / "convergence_M0.3_updated.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Separate Fast SIP Convergence for different Mach numbers
    plt.figure(figsize=(10, 6))
    for i, m in enumerate([0.1, 0.2, 0.3, 0.4]):
        plt.semilogy(fast_sip[m][0], fast_sip[m][1], color=colors[i], marker='o', linestyle='-', label=f'Fast SIP M={m} ({iters["FastSIP"][m]} iters)')
    
    plt.xlabel('Outer Iterations')
    plt.ylabel('Max Correction |delta_phi|')
    plt.title('Convergence History: Accelerated PNM (Fast SIP)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim(0, 120)
    plt.savefig(results_dir / "convergence_fast_only.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Updated convergence plots generated in results/")

    print("Updated convergence plots generated in results/")

if __name__ == "__main__":
    plot_convergence_from_data()
