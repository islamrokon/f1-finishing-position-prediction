#my three EDA figures for the group exploratory analysis.



#fig 1: driver-vs-constructor win rates, 3 panels. The conditional heatmap is

#fig 2: constructor strength by final position, split by era. This one carries
#the new-knowledge finding (V8 68% -> Hybrid 90% -> GE 74% wins from top-2).
#fig 3: PCA scree + biplot for the B-band unsupervised requirement.




import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_PATH = 'f1_final_dataset.csv'
SEED = 42

mpl.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})



NUMERIC_COLS = ['grid', 'q1_gap_to_pole',
                'constructor_standing', 'constructor_points', 'constructor_wins',
                'driver_standing', 'driver_points', 'driver_wins',
                'alt', 'year', 'round']




LABELS = ['grid', 'Q1 gap (s)',
          'constr. standing', 'constr. points', 'constr. wins',
          'driver standing', 'driver points', 'driver wins',
          'altitude', 'year', 'round']

ERA_ORDER = ['V8 (2005-2013)', 'Hybrid (2014-2021)', 'Ground Effect (2022-2025)']




def get_era(year):
    if year <= 2013:
        return 'V8 (2005-2013)'
    if year <= 2021:
        return 'Hybrid (2014-2021)'
    return 'Ground Effect (2022-2025)'


def load_data(path):
    df = pd.read_csv(path)
    df['era'] = df['year'].apply(get_era)
    print(f'loaded {len(df):,} driver-races, '
          f'{df["year"].min()}-{df["year"].max()}')
    return df






def figure_driver_strength(df, out_path):
    print('\n[1/3] driver strength three-panel')

    def winrate_by(col, n_top=12):
        grp = df.groupby(col)
        out = pd.DataFrame({
            'n_races': grp.size(),
            'wins': grp['final_position'].apply(
                lambda x: (x == '1st').sum()),
        }).reset_index()
        out['win_rate'] = out['wins'] / out['n_races'] * 100
        return out.head(n_top)

    ds = winrate_by('driver_standing', 12)
    cs = winrate_by('constructor_standing', 12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    #(a) win rate by driver standing
    ax1 = axes[0]
    x = ds['driver_standing'].astype(int)
    bars = ax1.bar(x, ds['win_rate'], color='#3498db',
                   edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, ds['win_rate']):
        if v > 0.3:
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                     f'{v:.1f}%', ha='center', fontsize=8.5)
    ax1.set_xlabel('Driver championship standing (pre-race)')
    ax1.set_ylabel('Win rate (%)')
    ax1.set_title('(a) Win rate by driver standing',
                  fontsize=10, loc='left')
    ax1.set_xticks(range(1, 13))
    ax1.set_ylim(0, 45)
    ax1.grid(axis='y', alpha=0.3)




    #(b) win rate by constructor standing - matched axes for direct compare
    ax2 = axes[1]
    x = cs['constructor_standing'].astype(int)
    bars = ax2.bar(x, cs['win_rate'], color='#e74c3c',
                   edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, cs['win_rate']):
        if v > 0.3:
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                     f'{v:.1f}%', ha='center', fontsize=8.5)
    ax2.set_xlabel('Constructor championship standing (pre-race)')
    ax2.set_ylabel('Win rate (%)')
    ax2.set_title('(b) Win rate by constructor standing',
                  fontsize=10, loc='left')
    ax2.set_xticks(range(1, 13))
    ax2.set_ylim(0, 45)
    ax2.grid(axis='y', alpha=0.3)





    #(c) conditional-win rate by driver tier given constructor tier
    ax3 = axes[2]
    df_local = df.copy()
    df_local['constructor_tier'] = pd.cut(
        df_local['constructor_standing'],
        bins=[0, 2, 5, 10, 30],
        labels=['Top-2', '3rd-5th', '6th-10th', '11th+'])
    df_local['driver_tier'] = pd.cut(
        df_local['driver_standing'],
        bins=[0, 2, 5, 10, 30],
        labels=['Top-2', '3rd-5th', '6th-10th', '11th+'])

    pivot_n = df_local.pivot_table(
        index='constructor_tier', columns='driver_tier',
        values='final_position', aggfunc='count', observed=False)
    pivot_w = df_local.pivot_table(
        index='constructor_tier', columns='driver_tier',
        values='final_position',
        aggfunc=lambda x: (x == '1st').mean() * 100, observed=False)



    #mask cells with too few obs to compute reliable rates
    mask = pivot_n < 30
    pivot_w_masked = pivot_w.where(~mask, np.nan)

    im = ax3.imshow(pivot_w_masked.values, cmap='Reds',
                    vmin=0, vmax=35, aspect='auto')
    ax3.set_xticks(range(len(pivot_w.columns)))
    ax3.set_yticks(range(len(pivot_w.index)))
    ax3.set_xticklabels(pivot_w.columns)
    ax3.set_yticklabels(pivot_w.index)
    ax3.set_xlabel('Driver standing tier')
    ax3.set_ylabel('Constructor standing tier')

    for i in range(len(pivot_w.index)):
        for j in range(len(pivot_w.columns)):
            v = pivot_w.iloc[i, j]
            n = pivot_n.iloc[i, j]
            if pd.isna(v) or n < 30:
                ax3.text(j, i,
                         f'n<30\n(n={int(n) if not pd.isna(n) else 0})',
                         ha='center', va='center',
                         fontsize=8, color='gray')
            else:
                color = 'white' if v > 18 else 'black'
                ax3.text(j, i, f'{v:.1f}%\n(n={int(n)})',
                         ha='center', va='center', fontsize=8.5,
                         color=color, fontweight='bold')

    plt.colorbar(im, ax=ax3, shrink=0.8, label='Win rate (%)')
    ax3.set_title('(c) Win rate conditional on driver x constructor tier',
                  fontsize=10, loc='left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}')



    #numbers I want for the report (same car, different driver finding)
    print(f'  Driver #1 win rate    : {ds.iloc[0]["win_rate"]:.1f}%')
    print(f'  Constructor #1 win rt : {cs.iloc[0]["win_rate"]:.1f}%')
    print(f'  Top-2 driver in Top-2 constr.   : '
          f'{pivot_w.iloc[0,0]:.1f}% (n={int(pivot_n.iloc[0,0])})')
    print(f'  3rd-5th driver in Top-2 constr. : '
          f'{pivot_w.iloc[0,1]:.1f}% (n={int(pivot_n.iloc[0,1])})')
    print(f'  6-10 driver in Top-2 constr.    : '
          f'{pivot_w.iloc[0,2]:.1f}% (n={int(pivot_n.iloc[0,2])})')


def figure_era_constructor(df, out_path):
    print('\n[2/3] constructor strength by final position, era-stratified')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    pos_order = ['1st', '2nd', '3rd', '4th', 'Outside Top 4']
    box_colors = ['#f1c40f', '#95a5a6', '#d35400', '#34495e', '#bdc3c7']

    for ax, era in zip(axes, ERA_ORDER):
        sub = df[df['era'] == era]
        data = [sub.loc[sub['final_position'] == p,
                        'constructor_standing'].values
                for p in pos_order]
        bp = ax.boxplot(data,
                        labels=['1st', '2nd', '3rd', '4th', 'Outside'],
                        patch_artist=True, widths=0.65,
                        medianprops={'color': 'black', 'linewidth': 1.3},
                        flierprops={'marker': 'o', 'markersize': 2,
                                    'alpha': 0.3})
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)

        ax.set_title(f'{era}\nn={len(sub):,} driver-races',
                     fontsize=10, loc='left')
        ax.set_xlabel('Final position')
        ax.invert_yaxis()  #
        ax.grid(axis='y', alpha=0.3)




        medians = [np.median(d) if len(d) else np.nan for d in data]
        for i, m in enumerate(medians):
            if not np.isnan(m):
                ax.text(i + 1, m, f' {m:.0f}', va='center', ha='left',
                        fontsize=8, fontweight='bold')

    axes[0].set_ylabel('Constructor championship standing\n'
                       '(lower = stronger team)')
    fig.suptitle('Constructor strength vs. race outcome '
                 'across regulatory eras',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}')




    #numbers for sections 3 and 7
    print('  median constructor standing of winners by era:')
    winners = df[df['final_position'] == '1st']
    for era in ERA_ORDER:
        s = winners[winners['era'] == era]
        print(f'    {era}: median={s["constructor_standing"].median():.1f} '
              f'(n={len(s)} wins)')
    print('  % wins from top-2 constructor by era:')
    for era in ERA_ORDER:
        s = winners[winners['era'] == era]
        pct = (s['constructor_standing'] <= 2).mean() * 100
        print(f'    {era}: {pct:.1f}%')


def figure_pca(df, out_path):
    print('\n[3/3] PCA scree + biplot')
    X = df[NUMERIC_COLS].values
    X_std = StandardScaler().fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_std)
    var_exp = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)

    print('  variance explained:')
    for i, (v, c) in enumerate(zip(var_exp, cum_var)):
        print(f'    PC{i+1}: {v:5.2f}%  cumulative {c:5.2f}%')

    is_top4 = (df['final_position'] != 'Outside Top 4').values.astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) scree + cumulative variance on twin axis
    x_pc = np.arange(1, len(var_exp) + 1)
    bars = ax1.bar(x_pc, var_exp, color='#3498db',
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_xlabel('Principal component')
    ax1.set_ylabel('Variance explained (%)', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(x_pc)
    ax1.set_title('(a) Scree plot - PCA on standardised predictors',
                  fontsize=10, loc='left')
    for bar, v in zip(bars, var_exp):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 f'{v:.1f}%', ha='center', fontsize=8.5)

    ax1b = ax1.twinx()
    ax1b.plot(x_pc, cum_var, marker='o', color='#e74c3c', linewidth=1.8)
    ax1b.axhline(80, linestyle='--', color='#e74c3c',
                 alpha=0.5, linewidth=0.8)
    ax1b.text(len(x_pc) - 0.3, 82, '80% threshold',
              fontsize=8, color='#e74c3c', ha='right')
    ax1b.set_ylabel('Cumulative variance (%)', color='#e74c3c')
    ax1b.tick_params(axis='y', labelcolor='#e74c3c')
    ax1b.set_ylim(0, 105)
    ax1b.spines['top'].set_visible(False)

    #(b) biplot: subsample 2000 points to keep the scatter readable
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(scores), size=2000, replace=False)
    cols_pts = ['#bdc3c7' if t == 0 else '#e74c3c'
                for t in is_top4[idx]]
    ax2.scatter(scores[idx, 0], scores[idx, 1],
                c=cols_pts, s=8, alpha=0.55, linewidth=0)

    #loadings as arrows. Manual nudges so labels in the dense PC1 cluster


    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    arrow_scale = 3.0
    label_nudge = {
        'grid':              (0.0, -0.35),
        'Q1 gap (s)':        (0.0, +0.30),
        'constr. standing':  (0.0, -0.55),
        'driver standing':   (0.0, +0.55),
        'constr. points':    (0.0, +0.30),
        'driver points':     (0.0, -0.30),
        'constr. wins':      (0.0, -0.55),
        'driver wins':       (0.0, +0.55),
        'year':              (0.0,  0.0),
        'altitude':          (0.0,  0.0),
        'round':             (0.0,  0.0),
    }

    for lab, xy in zip(LABELS, loadings):
        ax2.arrow(0, 0, xy[0] * arrow_scale, xy[1] * arrow_scale,
                  color='#2c3e50', width=0.015, head_width=0.12,
                  length_includes_head=True, alpha=0.9)
        nudge = label_nudge.get(lab, (0.0, 0.0))
        lx = xy[0] * arrow_scale * 1.18 + nudge[0]
        ly = xy[1] * arrow_scale * 1.18 + nudge[1]
        ax2.text(lx, ly, lab,
                 fontsize=8.5, ha='center', va='center',
                 fontweight='bold', color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white',
                           ec='none', alpha=0.7))

    ax2.axhline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax2.axvline(0, color='gray', linewidth=0.4, alpha=0.5)
    ax2.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)')
    ax2.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)')
    ax2.set_title('(b) Biplot - grey = outside Top 4, '
                  'red = Top 4 finishers\n'
                  '(2,000 driver-races sampled for readability)',
                  fontsize=10, loc='left')

    plt.tight_layout()   #
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}')



    #PC1-3 loadings for the PCA paragraph in section 3
    loadings_df = pd.DataFrame(pca.components_[:3].T,
                               columns=['PC1', 'PC2', 'PC3'],
                               index=NUMERIC_COLS)
    print('  loadings (first 3 PCs):')
    print(loadings_df.round(3).to_string())



def main():
    df = load_data(DATA_PATH)
    figure_driver_strength(df,    'fig_1_driver_strength.png')
    figure_era_constructor(df,    'fig_2_era_constructor.png')
    figure_pca(df,                'fig_3_pca.png')
    print('\nthree figures generated.')


if __name__ == '__main__':
    main()
