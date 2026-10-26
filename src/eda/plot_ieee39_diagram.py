"""
Generate high-resolution Publication-Quality Single-Line Diagram (SLD) 
for the IEEE 39-Bus New England 10-Generator Benchmark System.
Highlights experimental fault buses, generator terminals, and N-1 contingency lines.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def generate_ieee39_sld(output_path: str = 'reports/figures/ieee39_single_line_diagram.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 39-Bus Planar Coordinates (Standard IEEE 39-bus layout representation)
    pos = {
        # Transmission & Load Buses (1 - 29)
        1: (2.0, 1.5), 2: (4.0, 1.2), 3: (7.0, 1.8), 4: (4.0, 3.2), 5: (3.0, 4.2),
        6: (3.2, 5.8), 7: (2.2, 5.2), 8: (2.2, 4.0), 9: (1.0, 6.2), 10: (4.2, 7.5),
        11: (4.6, 6.5), 12: (4.8, 5.2), 13: (5.8, 5.2), 14: (5.6, 3.8), 15: (6.8, 4.5),
        16: (7.2, 3.6), 17: (8.2, 3.2), 18: (8.5, 2.0), 19: (9.5, 5.0), 20: (10.0, 4.0),
        21: (9.0, 6.5), 22: (8.8, 7.5), 23: (7.8, 7.5), 24: (7.0, 6.5), 25: (5.5, 1.8),
        26: (5.8, 2.8), 27: (6.5, 2.8), 28: (8.5, 4.2), 29: (9.2, 3.2),
        
        # Generator Buses (30 - 39)
        30: (1.0, 7.5),   # Gen 1 (Swing)
        31: (2.0, 0.5),   # Gen 2
        32: (3.2, 7.0),   # Gen 3
        33: (9.5, 8.2),   # Gen 4
        34: (8.5, 8.5),   # Gen 5
        35: (7.5, 8.5),   # Gen 6
        36: (6.8, 7.5),   # Gen 7
        37: (5.2, 2.0),   # Gen 8
        38: (9.8, 2.5),   # Gen 9
        39: (1.0, 2.5),   # Gen 10 (High Inertia H=42.0s)
    }
    
    # Transmission Lines & Transformers (From ANDES ieee39_full.xlsx)
    lines = [
        (1, 2), (1, 39), (2, 3), (2, 25), (3, 4), (3, 18), (4, 5), (4, 14),
        (5, 6), (5, 8), (6, 7), (6, 11), (7, 8), (8, 9), (9, 39), (10, 11),
        (10, 13), (10, 32), (12, 11), (12, 13), (13, 14), (14, 15), (15, 16),
        (16, 17), (16, 19), (16, 21), (16, 24), (17, 18), (17, 27), (19, 20),
        (19, 33), (20, 34), (21, 22), (22, 23), (22, 35), (23, 24), (23, 36),
        (25, 26), (25, 37), (26, 27), (26, 28), (28, 29), (29, 38),
        # Generator step-up transformer connections
        (2, 31), (6, 31), (10, 32), (19, 33), (20, 34), (22, 35), (23, 36),
        (25, 37), (29, 38), (30, 2)
    ]
    # Remove duplicate pairs and self-loops
    clean_lines = list(set([tuple(sorted([u, v])) for u, v in lines if u != v]))

    # Target sets for highlighting
    gen_buses = set(range(30, 40))
    fault_buses = {2, 14, 16, 26, 29, 39}
    held_out_buses = {29, 39}
    contingency_line = tuple(sorted([16, 17]))

    plt.figure(figsize=(15, 11), facecolor='#0B1120')
    ax = plt.gca()
    ax.set_facecolor('#0B1120')

    # Draw standard transmission lines
    for u, v in clean_lines:
        edge = tuple(sorted([u, v]))
        if edge == contingency_line:
            continue  # Drawn separately with distinct style
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        ax.plot(x_coords, y_coords, color='#475569', linewidth=1.5, zorder=1, alpha=0.8)

    # Draw Held-Out Contingency Line (Line 16-17)
    u_c, v_c = contingency_line
    ax.plot([pos[u_c][0], pos[v_c][0]], [pos[u_c][1], pos[v_c][1]], 
            color='#EF4444', linewidth=3.5, linestyle='--', zorder=2, label='Held-Out N-1 Outage (Line 16-17)')

    # Draw Bus Nodes
    for bus_id, (x, y) in pos.items():
        if bus_id in gen_buses:
            # Generator buses: Cyan double circle / ring
            ax.scatter(x, y, s=550, color='#0284C7', edgecolors='#38BDF8', linewidth=2.5, zorder=4)
            # Generator icon indicator symbol (sine wave inside)
            ax.text(x, y, f"G{bus_id-29}", color='#FFFFFF', fontsize=9.5, fontweight='bold',
                    ha='center', va='center', zorder=5)
            # Bus number label above
            ax.text(x, y + 0.35, f"Bus {bus_id}", color='#38BDF8', fontsize=8.5, fontweight='bold',
                    ha='center', va='bottom', zorder=5)
        elif bus_id in held_out_buses:
            # Unseen held-out fault buses: High-alert Crimson
            ax.scatter(x, y, s=320, color='#DC2626', edgecolors='#FCA5A5', linewidth=2.5, zorder=3)
            ax.text(x, y, str(bus_id), color='#FFFFFF', fontsize=9, fontweight='bold',
                    ha='center', va='center', zorder=5)
            ax.text(x, y - 0.35, f"Held-Out Bus {bus_id}", color='#EF4444', fontsize=8.5, fontweight='bold',
                    ha='center', va='top', zorder=5)
        elif bus_id in fault_buses:
            # Trained in-distribution fault buses: Amber
            ax.scatter(x, y, s=280, color='#D97706', edgecolors='#FCD34D', linewidth=2, zorder=3)
            ax.text(x, y, str(bus_id), color='#FFFFFF', fontsize=9, fontweight='bold',
                    ha='center', va='center', zorder=5)
        else:
            # Standard transmission load buses: Slate
            ax.scatter(x, y, s=180, color='#1E293B', edgecolors='#64748B', linewidth=1.5, zorder=3)
            ax.text(x, y, str(bus_id), color='#CBD5E1', fontsize=8, fontweight='500',
                    ha='center', va='center', zorder=5)

    # Callout Annotation for Bus 39 (Gen 10)
    ax.annotate(
        "CRITICAL GENERATOR 10\nTERMINAL BUS 39\n(H = 42.0 s • Severe Acceleration)",
        xy=pos[39], xytext=(pos[39][0] - 0.4, pos[39][1] - 1.2),
        arrowprops=dict(facecolor='#EF4444', edgecolor='#EF4444', arrowstyle='->', lw=2),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E293B', edgecolor='#EF4444', lw=1.5),
        color='#FCA5A5', fontsize=9, fontweight='bold', ha='center', zorder=6
    )

    # Callout Annotation for Central Switching Hub Bus 16
    ax.annotate(
        "MAJOR TIE HUB BUS 16\n(5-Way Transmission Junction)",
        xy=pos[16], xytext=(pos[16][0] + 0.3, pos[16][1] + 1.1),
        arrowprops=dict(facecolor='#D97706', edgecolor='#D97706', arrowstyle='->', lw=2),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E293B', edgecolor='#D97706', lw=1.5),
        color='#FDE68A', fontsize=8.5, fontweight='bold', ha='center', zorder=6
    )

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Generator Bus (Buses 30–39)',
               markerfacecolor='#0284C7', markeredgecolor='#38BDF8', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Standard Transmission/Load Bus (Buses 1–29)',
               markerfacecolor='#1E293B', markeredgecolor='#64748B', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Trained Fault Bus (Buses 2, 14, 16, 26)',
               markerfacecolor='#D97706', markeredgecolor='#FCD34D', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Held-Out Unseen Fault Bus (Buses 29, 39)',
               markerfacecolor='#DC2626', markeredgecolor='#FCA5A5', markersize=11),
        Line2D([0], [0], color='#EF4444', lw=3, linestyle='--', label='Held-Out N-1 Outage (Line 16-17)'),
        Line2D([0], [0], color='#475569', lw=1.5, label='Normal 345 kV Transmission Line')
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#0F172A', edgecolor='#334155',
              fontsize=9.5, labelcolor='#F1F5F9', framealpha=0.9, borderpad=1)

    # Title & Subtitle
    plt.title("IEEE 39-Bus New England Test System — Transmission Topology & Disturbance Protocol",
              fontsize=14, fontweight='bold', color='#F8FAFC', pad=25, fontfamily='sans-serif')
    plt.suptitle("National Institute of Technology (NIT) Raipur • EE Minor Project • Transient Stability Assessment",
                 fontsize=10.5, color='#94A3B8', y=0.92, fontfamily='sans-serif')

    # Watermark / Info Box
    info_text = (
        "System Parameters:\n"
        "• 39 Buses, 46 Lines, 10 Generators\n"
        "• Base Voltage: 345 kV | System Freq: 60 Hz\n"
        "• Tested Fault Inceptions: Buses 2, 14, 16, 26, 29, 39\n"
        "• Held-Out Generalization Splits: Buses 29 & 39, Line 16-17"
    )
    plt.figtext(0.13, 0.12, info_text, fontsize=8.5, color='#94A3B8', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#0F172A', edgecolor='#334155', alpha=0.9))

    ax.set_xlim(0, 11)
    ax.set_ylim(-0.2, 9.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0B1120')
    plt.close()
    print(f"IEEE 39-Bus Single-Line Diagram successfully saved to: {output_path}")

if __name__ == '__main__':
    generate_ieee39_sld()
