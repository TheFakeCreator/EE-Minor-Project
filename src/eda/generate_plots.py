import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Setup directories
    os.makedirs('src/eda', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load dataset
    csv_path = 'data/processed/dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} scenarios.")
    
    # Set plotting style
    sns.set_theme(style="whitegrid")
    
    # 1. Class Balance
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='is_stable', palette='Set2')
    ax.set_title('Class Balance: Stable (1) vs Unstable (0)')
    ax.set_xlabel('Is Stable')
    ax.set_ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig('reports/figures/class_balance.png', dpi=300)
    plt.close()
    
    # 2. Stability vs Clearing Time
    plt.figure(figsize=(8, 5))
    # Calculate instability rate (1 - mean(is_stable)) since is_stable=1 is stable
    ct_rates = df.groupby('clearing_time')['is_stable'].apply(lambda x: 1 - x.mean()).reset_index()
    sns.lineplot(data=ct_rates, x='clearing_time', y='is_stable', marker='o', linewidth=2)
    plt.title('Instability Probability vs. Fault Clearing Time')
    plt.xlabel('Clearing Time (s)')
    plt.ylabel('Probability of Instability')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('reports/figures/stability_vs_clearing_time.png', dpi=300)
    plt.close()
    
    # 3. Stability by Fault Bus
    plt.figure(figsize=(8, 5))
    bus_rates = df.groupby('fault_bus')['is_stable'].apply(lambda x: 1 - x.mean()).reset_index()
    bus_rates = bus_rates.sort_values('is_stable', ascending=False)
    # Convert bus to string for categorical plotting
    bus_rates['fault_bus'] = bus_rates['fault_bus'].astype(str)
    sns.barplot(data=bus_rates, x='fault_bus', y='is_stable', palette='Reds_r')
    plt.title('Instability Probability by Fault Bus')
    plt.xlabel('Fault Bus Number')
    plt.ylabel('Probability of Instability')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('reports/figures/stability_by_bus.png', dpi=300)
    plt.close()
    
    # 4. Feature Distributions (v_min_system)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='is_stable', y='v_min_system', palette='Set2')
    plt.title('Distribution of Minimum System Voltage by Stability Class')
    plt.xlabel('Is Stable')
    plt.ylabel('Minimum Voltage (p.u.)')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png', dpi=300)
    plt.close()
    
    print("EDA plots generated and saved to reports/figures/.")

if __name__ == '__main__':
    main()
