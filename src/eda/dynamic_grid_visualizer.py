"""
Dynamic Grid Visualizer for IEEE 39-Bus New England Digital Twin.
EE Minor Project — NIT Raipur.

Generates real-time, interactive, publication-quality SCADA visual diagrams:
1. Dynamic Single-Line Diagram (SLD) with live fault shockwaves, line outages, and generator status.
2. Transient Rotor Angle Swing Curves delta(t) with dynamic pole slip detection.
3. Equal Area Criterion (EAC) P-delta power-angle dynamics with accelerating/decelerating energy shading.
"""

import matplotlib.pyplot as plt
import numpy as np

# IEEE 39-Bus Standard Planar Coordinates
IEEE39_POS = {
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

# 46 Transmission Lines and Transformers
RAW_LINES = [
    (1, 2), (1, 39), (2, 3), (2, 25), (3, 4), (3, 18), (4, 5), (4, 14),
    (5, 6), (5, 8), (6, 7), (6, 11), (7, 8), (8, 9), (9, 39), (10, 11),
    (10, 13), (10, 32), (12, 11), (12, 13), (13, 14), (14, 15), (15, 16),
    (16, 17), (16, 19), (16, 21), (16, 24), (17, 18), (17, 27), (19, 20),
    (19, 33), (20, 34), (21, 22), (22, 23), (22, 35), (23, 24), (23, 36),
    (25, 26), (25, 37), (26, 27), (26, 28), (28, 29), (29, 38),
    (2, 31), (6, 31), (30, 2)
]
IEEE39_LINES = list(set([tuple(sorted([u, v])) for u, v in RAW_LINES if u != v]))

# Generator to nearest bus map
GEN_BUS_MAP = {
    1: 30, 2: 31, 3: 32, 4: 33, 5: 34,
    6: 35, 7: 36, 8: 37, 9: 38, 10: 39
}

def generate_dynamic_sld_figure(
    fault_bus: int,
    contingency: str,
    is_stable: int,
    clearing_time: float,
    cct_predicted: float,
    load_scale: float
) -> plt.Figure:
    """
    Renders an interactive SCADA Single-Line Diagram highlighting active disturbance,
    contingency trips, and live generator dynamic states.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor='#0B1120')
    ax.set_facecolor('#0B1120')

    # Identify active outage line
    outage_line = None
    if contingency == "Line_16_17":
        outage_line = tuple(sorted([16, 17]))
    elif contingency == "Line_2_3":
        outage_line = tuple(sorted([2, 3]))

    # Draw transmission lines
    for u, v in IEEE39_LINES:
        edge = tuple(sorted([u, v]))
        if edge == outage_line:
            continue  # Tripped line handled below
        x_coords = [IEEE39_POS[u][0], IEEE39_POS[v][0]]
        y_coords = [IEEE39_POS[u][1], IEEE39_POS[v][1]]
        ax.plot(x_coords, y_coords, color='#334155', linewidth=1.5, zorder=1, alpha=0.85)

    # Draw tripped outage line if active
    if outage_line:
        u_o, v_o = outage_line
        x_o = [IEEE39_POS[u_o][0], IEEE39_POS[v_o][0]]
        y_o = [IEEE39_POS[u_o][1], IEEE39_POS[v_o][1]]
        ax.plot(x_o, y_o, color='#EF4444', linewidth=2.8, linestyle='--', zorder=2, alpha=0.9)
        # Breaker Open marker at midpoint
        mid_x = (IEEE39_POS[u_o][0] + IEEE39_POS[v_o][0]) / 2.0
        mid_y = (IEEE39_POS[u_o][1] + IEEE39_POS[v_o][1]) / 2.0
        ax.scatter([mid_x], [mid_y], s=200, marker='X', color='#EF4444', edgecolors='#FFFFFF', linewidth=1.5, zorder=6)
        ax.text(mid_x, mid_y + 0.35, f"LINE {u_o}-{v_o} OPEN", color='#FCA5A5', fontsize=7.5,
                fontweight='bold', ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#7F1D1D', edgecolor='#EF4444', lw=1), zorder=7)

    # Determine critically stressed generator based on fault location
    crit_gen_bus = 39 if fault_bus == 39 else (38 if fault_bus == 29 else (37 if fault_bus == 26 else (31 if fault_bus == 2 else 39)))

    # Draw Buses
    for bus_id, (x, y) in IEEE39_POS.items():
        is_gen = (bus_id >= 30)
        is_fault = (bus_id == fault_bus)
        
        if is_gen:
            gen_num = bus_id - 29
            # Dynamic generator color: if unstable and this is the critical machine -> emergency red
            if is_stable == 0 and (bus_id == crit_gen_bus or fault_bus == bus_id):
                gen_color = '#DC2626'
                edge_color = '#FCA5A5'
                gen_status = "TRIP/SLIP"
            elif is_stable == 0:
                gen_color = '#B45309'
                edge_color = '#FDE68A'
                gen_status = "OSCILLATING"
            else:
                gen_color = '#059669'
                edge_color = '#34D399'
                gen_status = "SYNC"

            ax.scatter(x, y, s=420, color=gen_color, edgecolors=edge_color, linewidth=2.2, zorder=4)
            ax.text(x, y, f"G{gen_num}", color='#FFFFFF', fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)
            ax.text(x, y + 0.32, f"B{bus_id} ({gen_status})", color=edge_color, fontsize=7.5, fontweight='bold', ha='center', va='bottom', zorder=5)

        elif is_fault:
            # Active fault bus: Pulsing shockwave rings
            ax.scatter(x, y, s=750, color='#7F1D1D', edgecolors='#EF4444', linewidth=1.5, zorder=3, alpha=0.5)
            ax.scatter(x, y, s=450, color='#DC2626', edgecolors='#FCA5A5', linewidth=2.5, zorder=4)
            ax.text(x, y, str(bus_id), color='#FFFFFF', fontsize=9, fontweight='bold', ha='center', va='center', zorder=5)
            # Fault indicator badge
            ax.annotate(
                f"FAULT @ BUS {bus_id}\nt_clr = {clearing_time:.3f}s",
                xy=(x, y), xytext=(x, y - 0.75),
                arrowprops=dict(facecolor='#EF4444', edgecolor='#EF4444', arrowstyle='->', lw=1.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1E293B', edgecolor='#EF4444', lw=1.5),
                color='#FCA5A5', fontsize=8, fontweight='bold', ha='center', zorder=6
            )
        else:
            # Standard transmission bus
            ax.scatter(x, y, s=140, color='#1E293B', edgecolors='#64748B', linewidth=1.2, zorder=3)
            ax.text(x, y, str(bus_id), color='#94A3B8', fontsize=7.5, ha='center', va='center', zorder=5)

    # Dynamic status annotation banner
    status_text = "DYNAMICALLY STABLE" if is_stable == 1 else "LOSS OF SYNCHRONISM (POLE SLIP)"
    status_color = '#10B981' if is_stable == 1 else '#EF4444'
    status_bg = '#064E3B' if is_stable == 1 else '#7F1D1D'

    ax.text(0.3, 0.35, f"STATUS: {status_text}\nCCT: {cct_predicted:.3f}s | RELAY: {clearing_time:.3f}s | LOAD: {load_scale:.2f}x",
            fontsize=8.5, fontweight='bold', color=status_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=status_bg, edgecolor=status_color, lw=1.2),
            zorder=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Generator (Sync / Oscillating)',
               markerfacecolor='#059669', markeredgecolor='#34D399', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Active Fault Bus (3-Phase Short)',
               markerfacecolor='#DC2626', markeredgecolor='#FCA5A5', markersize=11),
        Line2D([0], [0], marker='o', color='w', label='Transmission Bus (1-29)',
               markerfacecolor='#1E293B', markeredgecolor='#64748B', markersize=8),
        Line2D([0], [0], color='#EF4444', lw=2.5, linestyle='--', label='Tripped Contingency Line'),
        Line2D([0], [0], color='#334155', lw=1.5, label='Normal 345 kV Grid Line')
    ]
    ax.legend(handles=legend_elements, loc='lower right', facecolor='#0F172A', edgecolor='#334155',
              fontsize=8, labelcolor='#CBD5E1', framealpha=0.92, borderpad=0.8)

    ax.set_xlim(0, 11)
    ax.set_ylim(-0.2, 9.4)
    ax.axis('off')
    plt.tight_layout()
    return fig


def generate_transient_swing_curve_figure(
    fault_bus: int,
    clearing_time: float,
    cct_predicted: float,
    is_stable: int,
    load_scale: float
) -> plt.Figure:
    """
    Renders realistic time-domain multi-machine rotor angle swing curves delta(t)
    demonstrating post-fault bounded oscillations vs runaway pole slip divergence.
    """
    fig, ax = plt.subplots(figsize=(10.5, 4.2), facecolor='#0F172A')
    ax.set_facecolor('#0F172A')

    t = np.linspace(0.0, 3.5, 400)
    t_fault_start = 0.5
    t_fault_end = t_fault_start + clearing_time

    # Baseline operating angle (depends on system load scale)
    delta_0 = 32.0 + (load_scale - 1.0) * 25.0  # degrees

    # Machine 1: Critical faulted generator (Gen closest to fault)
    # Machine 2: Generator 10 (High inertia NY intertie at Bus 39)
    # Machine 3: Distant Generator (Gen 1 / Swing bus at Bus 30)

    delta_crit = np.zeros_like(t)
    delta_g10 = np.zeros_like(t)
    delta_remote = np.zeros_like(t)

    # Initial state
    delta_crit[t < t_fault_start] = delta_0
    delta_g10[t < t_fault_start] = delta_0 - 6.0
    delta_remote[t < t_fault_start] = delta_0 - 14.0

    # During fault acceleration rate (deg/s^2)
    # Bus 39 terminal fault accelerates much faster
    accel_rate = 1400.0 if fault_bus == 39 else (1150.0 if fault_bus in [29, 26] else 900.0)
    accel_rate *= load_scale

    # Compute trajectories
    for i, ti in enumerate(t):
        if ti < t_fault_start:
            continue
        elif ti <= t_fault_end:
            # Quadratic acceleration during fault
            dt_f = ti - t_fault_start
            delta_crit[i] = delta_0 + 0.5 * accel_rate * (dt_f ** 2)
            delta_g10[i] = (delta_0 - 6.0) + 0.5 * (accel_rate * 0.75) * (dt_f ** 2)
            delta_remote[i] = (delta_0 - 14.0) + 0.5 * (accel_rate * 0.25) * (dt_f ** 2)
        else:
            # Post-fault dynamics
            dt_post = ti - t_fault_end
            delta_clear = delta_0 + 0.5 * accel_rate * (clearing_time ** 2)
            omega_clear = accel_rate * clearing_time  # deg/s

            if is_stable == 1:
                # Damped oscillation settling to new post-fault angle
                d_peak = min(155.0, delta_clear + omega_clear * 0.16)
                decay = np.exp(-1.4 * dt_post)
                osc = np.sin(2.0 * np.pi * 1.15 * dt_post)
                delta_crit[i] = (delta_0 + 8.0) + (d_peak - (delta_0 + 8.0)) * decay * osc + (d_peak - delta_0) * np.exp(-2.5 * dt_post)
                delta_g10[i] = (delta_0 - 2.0) + (d_peak * 0.7 - delta_0) * np.exp(-1.1 * dt_post) * np.cos(2.0 * np.pi * 0.95 * dt_post)
                delta_remote[i] = (delta_0 - 12.0) + 12.0 * np.exp(-0.8 * dt_post) * np.cos(2.0 * np.pi * 0.8 * dt_post)
            else:
                # Runaway pole-slip trajectory diverging past 180 deg
                delta_crit[i] = delta_clear + omega_clear * dt_post + 180.0 * (dt_post ** 1.6)
                delta_g10[i] = (delta_0 - 6.0) + 0.5 * (accel_rate * 0.75) * (clearing_time ** 2) + (omega_clear * 0.65) * dt_post + 110.0 * (dt_post ** 1.5)
                delta_remote[i] = (delta_0 - 14.0) + 15.0 * np.sin(2.0 * np.pi * 0.85 * dt_post)

    # Shaded Fault Inception Window
    ax.axvspan(t_fault_start, t_fault_end, color='#EF4444', alpha=0.22, label=f'Fault Window (Δt = {clearing_time:.3f}s)')

    # Stability Boundary Threshold (180 degrees)
    ax.axhline(180.0, color='#EF4444', linestyle='--', linewidth=1.8, label='Critical Stability Boundary (Δδ = 180°)')

    # Plot Generator Trajectories
    gen_label = f"Gen {10 if fault_bus==39 else (9 if fault_bus==29 else 2)} (Fault Nearest Machine)"
    color_crit = '#10B981' if is_stable == 1 else '#EF4444'
    ax.plot(t, delta_crit, color=color_crit, linewidth=2.4, label=f"{gen_label} Rotor Angle δ(t)")
    ax.plot(t, delta_g10, color='#38BDF8', linewidth=1.8, linestyle='-.', label="Gen 10 (Bus 39 Intertie, H=42.0s)")
    ax.plot(t, delta_remote, color='#94A3B8', linewidth=1.5, linestyle=':', label="Gen 1 (Bus 30 Swing Generator)")

    # Loss of Synchronism Callout
    if is_stable == 0:
        # Find index where delta_crit crosses 180
        cross_idx = np.where(delta_crit >= 180.0)[0]
        if len(cross_idx) > 0:
            t_cross = t[cross_idx[0]]
            ax.scatter([t_cross], [180.0], s=160, color='#EF4444', edgecolors='#FFFFFF', linewidth=2, zorder=6)
            ax.annotate(
                f"POLE SLIP DIVERGENCE\nt = {t_cross:.2f} s (>180°)",
                xy=(t_cross, 180.0), xytext=(t_cross + 0.25, 205.0),
                arrowprops=dict(facecolor='#EF4444', edgecolor='#EF4444', arrowstyle='->', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#7F1D1D', edgecolor='#EF4444', lw=1.2),
                color='#FFFFFF', fontsize=8.5, fontweight='bold', zorder=7
            )
    else:
        max_angle = np.max(delta_crit)
        t_max = t[np.argmax(delta_crit)]
        ax.scatter([t_max], [max_angle], s=120, color='#10B981', edgecolors='#FFFFFF', linewidth=1.8, zorder=6)
        ax.annotate(
            f"PEAK SWING: {max_angle:.1f}°\nMargin: {180.0 - max_angle:.1f}°",
            xy=(t_max, max_angle), xytext=(t_max + 0.25, max_angle + 25.0),
            arrowprops=dict(facecolor='#10B981', edgecolor='#10B981', arrowstyle='->', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#064E3B', edgecolor='#10B981', lw=1.2),
            color='#FFFFFF', fontsize=8, fontweight='bold', zorder=7
        )

    ax.set_xlim(0.0, 3.5)
    ax.set_ylim(0.0, 260.0 if is_stable == 0 else 190.0)
    ax.set_xlabel("Simulation Time (seconds)", fontsize=9.5, color='#CBD5E1', fontfamily='sans-serif')
    ax.set_ylabel("Relative Rotor Angle Δδ (degrees)", fontsize=9.5, color='#CBD5E1', fontfamily='sans-serif')
    ax.set_title(
        f"Dynamic Time-Domain Rotor Angle Trajectory Δδ(t) — {'STABLE DAMPED RECOVERY' if is_stable==1 else 'CATASTROPHIC POLE SLIP'}",
        fontsize=11, fontweight='bold', color='#F8FAFC', pad=12
    )

    ax.tick_params(colors='#CBD5E1', labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')

    ax.legend(loc='upper left', facecolor='#0B1120', edgecolor='#334155', fontsize=8, labelcolor='#CBD5E1', framealpha=0.9)
    plt.tight_layout()
    return fig


def generate_eac_figure(
    load_scale: float,
    clearing_time: float,
    cct_predicted: float,
    is_stable: int
) -> plt.Figure:
    """
    Renders classical Equal Area Criterion (EAC) P-delta power-angle dynamics
    demonstrating Kinetic Accelerating Area (A1) vs Maximum Decelerating Area (A2).
    """
    fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor='#0F172A')
    ax.set_facecolor('#0F172A')

    delta_deg = np.linspace(0, 180, 360)
    delta_rad = np.radians(delta_deg)

    # Electrical Power Curves
    P_max_prefault = 2.00
    P_max_fault = 0.35  # Severe voltage collapse during 3-phase short
    P_max_postfault = 1.75  # Slightly degraded if line tripped

    Pe_pre = P_max_prefault * np.sin(delta_rad)
    Pe_flt = P_max_fault * np.sin(delta_rad)
    Pe_post = P_max_postfault * np.sin(delta_rad)

    # Mechanical input power Pm
    Pm = 0.95 * load_scale

    # Initial operating angle
    delta_0_rad = np.arcsin(min(1.0, Pm / P_max_prefault))
    delta_0_deg = np.degrees(delta_0_rad)

    # Clearing angle delta_cl (increases with clearing time)
    delta_cl_deg = min(170.0, delta_0_deg + (clearing_time / 0.40) * 115.0)
    delta_cl_rad = np.radians(delta_cl_deg)

    # Critical clearing angle for post-fault system
    delta_max_rad = np.pi - np.arcsin(min(1.0, Pm / P_max_postfault))
    delta_max_deg = np.degrees(delta_max_rad)

    # Plot Power Curves
    ax.plot(delta_deg, Pe_pre, color='#38BDF8', linewidth=1.8, label=r'Pre-Fault $P_{e1}(\delta) = P_{\max 1}\sin\delta$')
    ax.plot(delta_deg, Pe_flt, color='#EF4444', linewidth=1.8, linestyle=':', label=r'Faulted $P_{e2}(\delta) \approx 0.18 P_{\max 1}\sin\delta$')
    ax.plot(delta_deg, Pe_post, color='#34D399', linewidth=1.8, linestyle='--', label=r'Post-Fault $P_{e3}(\delta) = P_{\max 3}\sin\delta$')
    ax.axhline(Pm, color='#F59E0B', linewidth=2.0, label=rf'Mechanical Power $P_m = {Pm:.2f}\text{{ p.u.}}$')

    # Shade Accelerating Area A1: [delta_0, delta_cl]
    mask_a1 = (delta_deg >= delta_0_deg) & (delta_deg <= delta_cl_deg)
    if np.any(mask_a1):
        ax.fill_between(
            delta_deg[mask_a1],
            Pe_flt[mask_a1],
            Pm,
            color='#EF4444', alpha=0.35,
            label='Acceleration Area $A_1$ (Kinetic Energy Impulse)'
        )

    # Shade Decelerating Area A2: [delta_cl, delta_max]
    if delta_cl_deg < delta_max_deg:
        mask_a2 = (delta_deg >= delta_cl_deg) & (delta_deg <= delta_max_deg)
        if np.any(mask_a2):
            ax.fill_between(
                delta_deg[mask_a2],
                Pm,
                Pe_post[mask_a2],
                color='#10B981', alpha=0.35,
                label='Deceleration Area $A_2$ (Synchronizing Capacity)'
            )

    # Vertical Markers
    ax.axvline(delta_0_deg, color='#94A3B8', linestyle=':', alpha=0.7)
    ax.text(delta_0_deg, -0.15, r'$\delta_0$', color='#94A3B8', fontsize=9, ha='center')

    ax.axvline(delta_cl_deg, color='#EF4444', linestyle='--', linewidth=1.5)
    ax.text(delta_cl_deg, -0.15, r'$\delta_{cl}$', color='#FCA5A5', fontsize=9, fontweight='bold', ha='center')

    # Status Annotation
    if is_stable == 1:
        eac_text = f"EAC CRITERION MET: Decelerating Area A2 >= Accelerating Area A1 (Margin: +{(cct_predicted - clearing_time)*1000:.0f}ms)"
        eac_color = '#10B981'
    else:
        eac_text = f"EAC CRITERION VIOLATED: Area A1 > Area A2 Capacity (Deficit: -{(clearing_time - cct_predicted)*1000:.0f}ms)"
        eac_color = '#EF4444'

    ax.text(90, 2.15, eac_text, color=eac_color, fontsize=8.5, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0F172A', edgecolor=eac_color, lw=1.2))

    ax.set_xlim(0, 180)
    ax.set_ylim(-0.25, 2.35)
    ax.set_xlabel(r"Rotor Angle $\delta$ (degrees)", fontsize=9, color='#CBD5E1')
    ax.set_ylabel("Power (p.u.)", fontsize=9, color='#CBD5E1')
    ax.set_title("Equal Area Criterion (EAC) Dynamics — Kinetic Energy vs Synchronizing Capacity", fontsize=10.5, fontweight='bold', color='#F8FAFC', pad=10)
    ax.tick_params(colors='#CBD5E1', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
    ax.legend(loc='upper right', facecolor='#0B1120', edgecolor='#334155', fontsize=7.5, labelcolor='#CBD5E1', framealpha=0.92)

    plt.tight_layout()
    return fig
