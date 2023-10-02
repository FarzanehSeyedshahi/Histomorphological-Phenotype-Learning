# stop warnings
import warnings
warnings.filterwarnings("ignore")


import os
import pandas as pd
import numpy as np
import umap
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def get_umap(path, fold_num, resolution, kernel):
    csv_complete_path_pred_test = os.path.join(path, 'kernel-{}_res-{}_fold-{}_test_pred.csv'.format(kernel, resolution, fold_num))
    # csv_complete_path_pred_train = os.path.join(path, 'kernel-{}_res-{}_fold-{}_train_pred.csv'.format(kernel, resolution, fold_num))
    umap_path = os.path.join(path, 'plots', 'umaps')

    # df_train = pd.read_csv(csv_complete_path_pred_train)
    df_test = pd.read_csv(csv_complete_path_pred_test)
    data = df_test.drop(['y_pred','y_probas_0','y_probas_1','y_true'], axis=1).to_numpy()


    # Plotly interactive plot
    umap_result = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(data)


    umap_df = pd.DataFrame(umap_result, columns=['UMAP 1', 'UMAP 2'])
    umap_df['Data Index'] = np.arange(df_test.index.tolist().size)

    fig = px.scatter(umap_df, x='UMAP 1', y='UMAP 2', hover_data=['Data Index'])
    fig.update_layout(title='Interactive UMAP Plot')
    # save plot as html file
    fig.write_html(os.path.join(umap_path, 'umap_interactive.html'))
    fig.show()


# Representations and Cluster Network.
def show_umap_leiden(adata, meta_field, layout, random_state, threshold, node_size_scale, node_size_power, edge_width_scale, directory, file_name,
                     fontsize=10, fontoutline=2, marker_size=2, ax_size=16, l_size=12, l_t_size=14, l_box_w=1, l_markerscale=1, palette='tab20', figsize=(30,10),
                     leiden_name=False):
    from matplotlib.lines import Line2D

    leiden_clusters = np.unique(adata.obs[groupby].astype(int)) # Leiden clusters.[0,...44]
    colors = sns.color_palette(palette, len(leiden_clusters))

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 3, 1)
    print(meta_field)

    ax = sc.pl.umap(adata, ax=ax, color=meta_field, size=marker_size, show=False, frameon=False, na_color='black')
    if meta_field == 'Meso_type':
        legend_c = ax.legend(loc='best', markerscale=l_markerscale, title='Meso Type', prop={'size': l_size})
        legend_c.get_title().set_fontsize(l_t_size)
        legend_c.get_frame().set_linewidth(l_box_w)
        legend_c.get_texts()[0].set_text('Epithelioid(0)')
        legend_c.get_texts()[1].set_text('Sarcomatoid(1)')
    ax.set_title('Tile Vector\nRepresentations', fontsize=ax_size, fontweight='bold')

    ax  = fig.add_subplot(1, 3, 2)
    sc.pl.umap(adata, ax=ax, color=groupby, size=marker_size, show=False, legend_loc='on data', legend_fontsize=fontsize, legend_fontoutline=fontoutline, frameon=False, palette=colors)
    if leiden_name:
        ax.set_title('Leiden Clusters', fontsize=ax_size, fontweight='bold')
    else:
        ax.set_title('Histomorphological Phenotype\nClusters', fontsize=ax_size, fontweight='bold')

    adjust_text(ax.texts)

    ax  = fig.add_subplot(1, 3, 3)
    names_lines = ['Epithelioid', 'Sarcomatoid']
    sc.pl.paga(adata, layout=layout, random_state=random_state, color=meta_field, threshold=threshold, node_size_scale=node_size_scale, node_size_power=node_size_power, edge_width_scale=edge_width_scale, fontsize=fontsize, fontoutline=fontoutline, frameon=False, show=False, ax=ax)
    if meta_field == 'Meso_type':
        legend = ax.legend(legend_c.legendHandles, names_lines, title='Meso Type', loc='upper left', prop={'size': l_size})
        legend.get_title().set_fontsize(l_t_size)
        legend.get_frame().set_linewidth(l_box_w)
    if leiden_name:
        ax.set_title('Leiden Cluster Network', fontsize=ax_size, fontweight='bold')
    else:
        ax.set_title('Histomorphological Phenotype\nCluster Network', fontsize=ax_size, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,file_name))
    plt.show()


if __name__ == '__main__':
    kernel = 'rbf'
    resolution = 2.0
    fold_num = 0


    pred_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning/datasets/clf_csvs/preds/'
    

    

    # draw umap
    get_umap(pred_path, fold_num, resolution, kernel)
