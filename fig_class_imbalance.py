#Class imbalance

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


DATA_PATH = 'f1_final_dataset.csv'
OUT_PATH  = 'class_imbalance.png'

mpl.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

CLASS_ORDER = ['1st', '2nd', '3rd', '4th', 'Outside Top 4']

#heat to cool ramp for podium positions, neutral grey for the majority class
COLORS = ['#c0392b', '#e67e22', '#f1c40f', '#16a085', '#bdc3c7']


def main():
    df = pd.read_csv(DATA_PATH)
    n_total = len(df)
    counts = df['final_position'].value_counts().reindex(CLASS_ORDER)
    pcts = counts / n_total * 100

    print(f'total observations: {n_total:,}')
    for cls, n, pct in zip(CLASS_ORDER, counts.values, pcts.values):
        print(f'  {cls:<15} n={n:>5,}  ({pct:5.2f}%)')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(CLASS_ORDER, counts.values, color=COLORS,
                  edgecolor='black', linewidth=0.6)

    for bar, n, pct in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + n_total * 0.012,
                f'{pct:.1f}%\n(n={n:,})',
                ha='center', va='bottom', fontsize=9.5)

#

    ax.set_xlabel('Finishing position class')
    ax.set_ylabel('Number of driver-races')
    ax.set_title(f'Class distribution of target variable '
                 f'(final_position)\n'
                 f'{n_total:,} driver-races, 2005-2025',
                 fontsize=11, loc='left')
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis='y', alpha=0.3)




    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'\nsaved {OUT_PATH}')


if __name__ == '__main__':
    main()
