import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import wandb
import pickle
import argparse

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from time import time


parser = argparse.ArgumentParser(description='Report classification and cluster performance based on different classifications.')
parser.add_argument('--mode',         dest='mode',         type=str,            default='train',        help='test or train model.')
parser.add_argument('--wandb',         dest='wandb_tracking',         type=bool,            default=False,        help='wandb tracking.')
parser.add_argument('--save_model',         dest='save_model',         type=bool,            default=False,        help='save model.')
parser.add_argument('--save_pred',         dest='save_pred',         type=bool,            default=False,        help='save predictions.')
parser.add_argument('--resolutions',         dest='resolutions',         type=list,            default=[2.0,5.0,7.0],        help='resolutions.')
parser.add_argument('--folds',         dest='folds',         type=list,            default=[0,1,2,3],        help='folds.')
parser.add_argument('--kernels',         dest='kernels',         type=list,            default=['rbf'],        help='Classification kernel.')
parser.add_argument('--alpha',         dest='alpha',         type=float,            default=1.0,        help='Classification kernel.')
parser.add_argument('--model_name',         dest='model_name',         type=list,            default=['SVC', 'LogisticRegression'],        help='Classification model name.')
parser.add_argument('--dataset',         dest='dataset',         type=str,            default='Meso_400_subsampled',        help='Classification dataset name.')
parser.add_argument('--additional_dataset',         dest='additional_dataset',         type=str,            default='TCGA_MESO',        help='Additional classification dataset name.')
parser.add_argument('--meta_folder', type=str, default='meso_subtypes_nn400', help='meta data folder')


args			   = parser.parse_args()


def plotly_roc(clf, X, y):
    import plotly.express as px
    y_score = clf.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_score)
    fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate')    )
    
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    wandb.log({"Plotly ROC": fig})



def wandb_log_sklearn(clf, X, y):
    wandb.init(project="meso_subtype", name="Lr{}".format(time()))
    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)
    accuracy = accuracy_score(y_pred, y)
    # wandb.log({'accuracy': accuracy})
    wandb.summary["accuracy"] = accuracy

    # Visualize single plot
    wandb.sklearn.plot_confusion_matrix(y, y_pred, clf.classes_)
    wandb.sklearn.plot_roc(y, y_score)
    plotly_roc(clf, X, y)
    
    wandb.sklearn.plot_precision_recall(y, clf.predict_proba(X))
    # wandb.sklearn.plot_class_prc(y, clf.predict_proba(X))
    # wandb.sklearn.plot_summary_metrics(clf, X, y)
    wandb.sklearn.plot_learning_curve(clf, X, y)
    wandb.sklearn.plot_calibration_curve(clf, X, y)
    wandb.sklearn.plot_feature_importances(clf, X, y)
    wandb.sklearn.plot_learning_curve(clf, X, y)

    print(classification_report(y, y_pred))


def umap_plot(X,y, title='UMAP projection of the dataset'):
    # draw umap from X and color it based on y
    import umap
    import plotly.express as px
    reducer = umap.UMAP()
    print('UMAP embedding...')
    embedding = reducer.fit_transform(X)
    fig = px.scatter(
        embedding, x=0, y=1, color=y,
        title='UMAP projection of the dataset', labels={'color': 'digit'}
    )
    # wandb.log({"UMAP": fig})
    fig.show()
    fig.write_html('umap_{}.html'.format(title))
    print('UMAP embedding saved as umap_{}.png'.format(title))


def pca_plot(X,y, title='PCA projection of the dataset'):
    # draw umap from X and color it based on y
    from sklearn.decomposition import PCA
    import plotly.express as px
    reducer = PCA(n_components=2)
    print('PCA embedding...')
    embedding = reducer.fit_transform(X)
    fig = px.scatter(
        embedding, x=0, y=1, color=y,
        title='PCA projection of the dataset', labels={'color': 'digit'}
    )
    # wandb.log({"PCA": fig})
    fig.show()
    fig.write_html('pca_{}.html'.format(title))
    print('PCA embedding saved as pca_{}.html'.format(title))


def correlation_matrix(X, title='Correlation matrix'):
    # draw correlation matrix from df
    import plotly.express as px
    df = pd.DataFrame(X)
    # print(df)
    # corr = df.corr()
    # fig = px.imshow(corr)
    # fig.show()
    # fig.write_html('corr_{}.html'.format(title))
    # print('Correlation matrix saved as corr_{}.png'.format(title))

    # heatmap
    import seaborn as sns
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, cmap='coolwarm')
    plt.savefig('patient_hpc_{}_percentage.png'.format(title))
    plt.show()

def forest_plot():
    # this link:https://medium.com/@ginoasuncion/visualizing-logistic-regression-results-using-a-forest-plot-in-python-bc7ba65b55bb
		pass  

if __name__ == '__main__':
    main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
    mode = args.mode
    wandb_tracking = args.wandb_tracking
    save_model = args.save_model
    save_pred = args.save_pred
    resolutions = args.resolutions
    dataset = args.dataset
    additional_dataset = args.additional_dataset
    meta_folder = args.meta_folder
    type_composition = 'clr'
    csv_paths = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128_filtered/{}/subtype_csvs/'.format(main_path, dataset, meta_folder)
    folds = args.folds
    alpha = args.alpha
    kernels = ['liblinear']
    # models = ['SVC', 'LogisticRegression', 'RandomForestClassifier']
    models = args.model_name
    for model in models:
        print('_'*100, '\n')
        print('Model:', model, '\n','_'*100)
        for resolution in resolutions:
            print('Resolution:', resolution)
            for fold in folds:
                groupby= 'leiden_%s' % resolution
                data_df = pd.read_csv(csv_paths + '/{}_{}_{}_fold{}.csv'.format(dataset, type_composition, groupby.replace('.', 'p'), fold), index_col=0)
                additional_path = csv_paths + '/{}_{}_{}_fold{}_additional.csv'.format(additional_dataset, type_composition, groupby.replace('.', 'p'), fold)
                if os.path.exists(additional_path):
                    data_df_additional = pd.read_csv(additional_path, index_col=0)
                    data_df_additional['Meso_type_x'] = data_df_additional['type'].apply(lambda x: 0 if x == 'Epithelioid' else 1)

                meso_col_ind = data_df.columns.get_loc('Meso_type_x')
                print('_'*50)
                print('Number of clusters:', meso_col_ind-2)
                X, y = data_df.iloc[:, 1:meso_col_ind].values, data_df.iloc[:, meso_col_ind].values
                # umap_plot(X,y, title='Training dataset')
                # correlation_matrix(X, title='Training dataset')


                from sklearn.preprocessing import StandardScaler
                # scaler = StandardScaler()
                # X = scaler.fit_transform(X)



                # upsample minority class
                from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE # SMOTE
                from imblearn.combine import SMOTEENN, SMOTETomek # SMOTETomek
                sm = SMOTETomek(random_state=42)
                X, y = sm.fit_resample(X, y)


                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                for k in kernels:
                    if model == 'SVC':
                        clf = SVC(kernel='rbf', probability=True, C=alpha).fit(X_train, y_train)
                    elif model == 'LogisticRegression':
                        clf = LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', C=alpha).fit(X_train, y_train)
                    else:
                        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train, y_train)
                        
                    y_pred = clf.predict(X_test)
                    print('Resolution {} - Fold {}'.format(resolution, fold))
                    # wandb_log_sklearn(clf, X_test, y_test)
                    if wandb_tracking:
                        wandb.init(project="meso_subtype", name=model)
                        wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred=y_pred, y_probas=clf.predict_proba(X_test), labels=clf.classes_, model_name='Logistic Regression')
                        wandb.summary["accuracy"] = accuracy_score(y_pred, y_test)
                        wandb.summary["f1_score"] = f1_score(y_pred, y_test)
                        wandb.summary["precision"] = average_precision_score(y_pred, y_test)
                        wandb.finish()
                    print(classification_report(y_test, y_pred))
                    print(average_precision_score(y_pred, y_test))


                if os.path.exists(additional_path):
                    X_additional, y_additional = data_df_additional.iloc[:, 1:meso_col_ind].values, data_df_additional.iloc[:, meso_col_ind].values
                    X_additional, y_additional = sm.fit_resample(X_additional, y_additional)
                    # umap_plot(X_additional,y_additional, title='Additional dataset')
                    # correlation_matrix(X_additional, title='Additional dataset')
                    # X_additional = scaler.transform(X_additional)
                    print('Resolution {} - Fold {} (additional dataset)'.format(resolution, fold))
                    y_pred_additional = clf.predict(X_additional)
                    print(classification_report(y_additional, y_pred_additional))
                    print(average_precision_score(y_additional, y_pred_additional))

                # X_total = np.concatenate((X, X_additional), axis=0)
                # y_temp = np.array(['train']*len(X))
                # y_additional_temp = np.array(['additional']*len(X_additional))
                # y_total = np.concatenate((y_temp, y_additional_temp ), axis=0)
                # umap_plot(X_total, y_total, title='Training and additional dataset')
                # pca_plot(X_total, y_total, title='Training and additional dataset')