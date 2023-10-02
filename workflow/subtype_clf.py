import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os
import sys
import anndata
import scanpy as sc
from adjustText import adjust_text
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
os.system('export CUDA_VISIBLE_DEVICES=4')
import wandb
import pickle
import argparse

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from time import time

main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
sys.path.append(main_path)
from models.clustering.correlations import *
from models.clustering.data_processing import *
from models.visualization.attention_maps import *
from data_manipulation.data import Data
from models.clustering.logistic_regression_leiden_clusters import *


# add arguments for mode



parser = argparse.ArgumentParser(description='Report classification and cluster performance based on different classifications.')
parser.add_argument('--mode',         dest='mode',         type=str,            default='test',        help='test or train model.')
parser.add_argument('--wandb',         dest='wandb_tracking',         type=bool,            default=False,        help='wandb tracking.')
parser.add_argument('--save_model',         dest='save_model',         type=bool,            default=False,        help='save model.')
parser.add_argument('--save_pred',         dest='save_pred',         type=bool,            default=False,        help='save predictions.')
parser.add_argument('--resolutions',         dest='resolutions',         type=list,            default=[2.0, 5.0, 7.0],        help='resolutions.')
parser.add_argument('--fold',         dest='fold',         type=int,            default=0,        help='fold.')
parser.add_argument('--kernel',         dest='kernel',         type=list,            default=['rbf'],        help='Classification kernel.')
parser.add_argument('--alpha',         dest='alpha',         type=float,            default=1.0,        help='Classification kernel.')
parser.add_argument('--model_name',         dest='model_name',         type=str,            default='SVC',        help='Classification model name.')


args			   = parser.parse_args()

def get_csvs_from_clusters(clusters, matching_field, groupby, i, fold, h5_complete_path, h5_additional_path, additional_as_fold=False, force_fold=None):
	
	return True

def model_confusion_matrix(model, data, label):
	train, valid, test, additional = data
	test_data,  test_labels  = test
	if additional is not None:
		additional_data, additional_labels = additional

	test_pred  = model.predict(test_data)
	if additional is not None:
		additional_pred = model.predict(additional_data)

	test_labels = test_labels[:,label]
	test_pred   = (test_pred > 0.5)*1.0

	cm_test = confusion_matrix(test_labels, test_pred)
	cm_additional = None
	if additional is not None:
		additional_labels = additional_labels[:,label]
		additional_pred   = (additional_pred > 0.5)*1.0
		cm_additional = confusion_matrix(additional_labels, additional_pred)
	return [cm_test, cm_additional]


def classification_performance(data, leiden_clusters, frame_clusters, features, groupby, alpha):
	train, valid, test, additional = data
	train_data, train_labels = train

	labels = np.unique(np.argmax(train_labels, axis=1)).tolist()
	if len(labels) == 2:
		labels.remove(0)

	# One-vs-rest.
	num_sets = len([1 for set in data if set is not None])
	shape_aucs = (len(labels), num_sets)
	total_aucs = np.zeros(shape_aucs)
	cms        = dict()
	for label in labels:

		model = SVC(kernel='rbf', C=1/alpha, class_weight='balanced').fit(train_data, train_labels[:,label])
		total_aucs[label-1,:]  = get_model_report(model, data, label)

		# Include information in Clusters DataFrame.
		# frame_clusters = include_coefficients(model, frame_clusters, features, label, groupby)
		frame_clusters = []

		label_cms = model_confusion_matrix(model, data, label)
		cms[label] = label_cms

	aucs = total_aucs.mean(axis=0).tolist()
	return frame_clusters, aucs, cms



def get_model_report(model, data, label):
	train, valid, test, additional = data
	train_data, train_labels = train	
	test_data,  test_labels  = test
	train_pred = model.predict(train_data)
	
	if valid is not None:
		valid_data, valid_labels = valid
		valid_pred = model.predict(valid_data)
	test_pred  = model.predict(test_data)
	if additional is not None:
		additional_data, additional_labels = additional
		additional_pred = model.predict(additional_data)

	print('train report:\n', classification_report(list(train_labels[:,label]), list(train_pred), labels=[0,1]))
	train_auc = roc_auc_score(y_true=list(train_labels[:,label]), y_score=list(train_pred))

	aucs = [train_auc]
	valid_auc = None
	if valid is not None:
		print('valid report:\n', classification_report(list(valid_labels[:,label]), list(valid_pred), labels=[0,1]))
		valid_auc = roc_auc_score(y_true=list(valid_labels[:,label]), y_score=list(valid_pred))
		aucs.append(valid_auc)

	test_auc  = roc_auc_score(y_true=list(test_labels[:,label]),  y_score=list(test_pred))
	print('test report:\n', classification_report(list(test_labels[:,label]), list(test_pred), labels=[0,1]))
	aucs.append(test_auc)
	
	additional_auc = None
	if additional is not None:
		additional_auc  = roc_auc_score(y_true=list(additional_labels[:,label]),   y_score=list(additional_pred))
		aucs.append(additional_auc)
	return aucs


def clf_report(model, data, mode='test'):
	train, valid, test, additional = data
	if mode == 'test':
		# Just for warwick data.
		X_train, y_train = train
		y_train = y_train[:,1]
		model = model.fit(X_train, y_train)
		#


		X, y = test
		y = y[:,1]
		y_pred = model.predict(X)
		y_probas = model.predict_proba(X)
		print('test report:\n', classification_report(list(y), list(y_pred), labels=[0,1]))
		print('test auc:\n', roc_auc_score(y_true=list(y), y_score=list(y_pred)))
		print('test confusion matrix:\n', confusion_matrix(y_true=list(y), y_pred=list(y_pred)))
		print('test average precision:\n', average_precision_score(y_true=list(y), y_score=list(y_pred)))

def loading_leiden_data(resolutions, h5_complete_path):
	folds_pickle_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning/MesoGraph/files/pkl_folds.pkl'
	# folds_pickle = pickle.load(open(folds_pickle_path, 'rb'))
	folds = load_existing_split(folds_pickle_path)

	# Path for alpha Logistic Regression results.
	meta_folder = 'Meso_nn250_tiles'
	meta_field = "meso_type"
	matching_field = "slides"
	h5_additional_path = None
	top_variance_feat = 99
	use_conn = False
	use_ratio = False
	min_tiles = 10
	additional_as_fold = False
	force_fold = None
	type_composition = 'clr'

	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)
	adatas_path  = os.path.join(main_cluster_path, 'adatas')


	data_res_folds = dict()
	for resolution in resolutions:
		groupby = 'leiden_%s' % resolution
		print('\tResolution', groupby)
		data_res_folds[resolution] = dict()
		for i, fold in enumerate(folds):

			# Read CSV files for train, validation, test, and additional sets.
			dataframes, complete_df, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, i, fold, h5_complete_path, h5_additional_path, additional_as_fold=additional_as_fold, force_fold=force_fold)
			train_df, valid_df, test_df, additional_df = dataframes

			# Check clusters and diversity within.
			frame_clusters, frame_samples = create_frames(complete_df, groupby, meta_field, diversity_key=matching_field, reduction=2)

			# Create representations per sample: cluster % of total sample.
			data, data_df, features = prepare_data_classes(dataframes, matching_field, meta_field, groupby, leiden_clusters, type_composition, min_tiles,
														use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat)
	
			# Include features that are not the regular leiden clusters.
			frame_clusters = include_features_frame_clusters(frame_clusters, leiden_clusters, features, groupby)

			# Store representations.
			data_res_folds[resolution][i] = {'data':data, 'features':features, 'frame_clusters':frame_clusters, 'leiden_clusters':leiden_clusters}
			# Information.
			print('\t\tFold', i, 'Features:', len(features), 'Clusters:', len(leiden_clusters))
	return data_res_folds

def wandb_tracking(clf, X_train, y_train, X_test, y_test, labels, resolution, fold_num, kernel, model_name='SVC'):
	wandb.init(project="meso_subtype", group="{}_Resolution-{}".format(model_name, resolution), name="{}_fold-{}_timestamp{}".format(kernel, fold_num, time()), entity='farzaneh_uog')
	wandb.sklearn.plot_feature_importances(clf, feature_names=None, max_num_features=20)
	wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
	wandb.sklearn.plot_confusion_matrix(y_pred, y_test, labels)
	wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
	wandb.sklearn.plot_roc(y_test, y_probas, labels)
	wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
	wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels, model_name=model_name, feature_names=None)

	train_auc = roc_auc_score(y_true=list(y_train), y_score=list(clf.predict(X_train)))
	test_auc  = roc_auc_score(y_true=list(y_test),  y_score=list(clf.predict(X_test)))
	wandb.log({'train_auc': train_auc, 'test_auc': test_auc})
	wandb.finish()

def save_predictions_csv(clf, X_test, y_test, X_train, y_train, resolution, fold_num, kernel, pred_path):
	df_prediction_test = pd.DataFrame(data=X_test, columns=np.arange(0,X_test.shape[1]))
	df_prediction_test['y_pred'] = clf.predict(X_test)
	df_prediction_test['y_probas_0'] = clf.predict_proba(X_test)[:,0]
	df_prediction_test['y_probas_1'] = clf.predict_proba(X_test)[:,1]
	df_prediction_test['y_true'] = y_test
	csv_complete_path_pred_test = os.path.join(pred_path, 'kernel-{}_res-{}_fold-{}_test_pred.csv'.format(kernel, resolution, fold_num))
	df_prediction_test.to_csv(csv_complete_path_pred_test)
	print('test prediction saved at {}'.format(csv_complete_path_pred_test))

	# for training data
	df_prediction_train = pd.DataFrame(data=X_train, columns=np.arange(0,X_train.shape[1]))
	df_prediction_train['y_pred'] = clf.predict(X_train)
	df_prediction_train['y_probas_0'] = clf.predict_proba(X_train)[:,0]
	df_prediction_train['y_probas_1'] = clf.predict_proba(X_train)[:,1]
	df_prediction_train['y_true'] = y_train
	csv_complete_path_pred_train = os.path.join(pred_path,'kernel-{}_res-{}_fold-{}_train_pred.csv'.format(kernel, resolution, fold_num))
	df_prediction_train.to_csv(csv_complete_path_pred_train)
	print('train prediction saved at {}'.format(csv_complete_path_pred_train))

if __name__ == '__main__':
	mode = args.mode
	wandb_tracking = args.wandb_tracking
	save_model = args.save_model
	save_pred = args.save_pred
	resolutions = args.resolutions
	if mode == 'train':
		i = args.fold
		
		alpha = args.alpha
		labels = [0,1]
		# kernels = ['rbf', 'poly', 'linear', 'sigmoid']
		kernels = args.kernel
		for resolution in resolutions:
			csv_complete_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning/results_250/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/meso_nn250/subtype_clf/clf_csvs/data_label_res-{}_fold-{}.csv'.format(resolution, i)
			data_df = pd.read_csv(csv_complete_path)
			X, y = data_df.iloc[:, :-1].values, data_df.iloc[:, -1].values
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			for k in kernels:
				clf = SVC(probability=True, kernel=k, class_weight='balanced').fit(X_train, y_train)
				if save_model:
					clf_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning/MesoGraph/models/clf_models/SVC_res-{}_fold-{}_kernel-{}.pkl'.format(resolution, i, k)
					with open(clf_path, 'wb') as file:
						pickle.dump(clf, file)
						print('model saved at {}'.format(clf_path))
				y_pred = clf.predict(X_test)
				y_probas = clf.predict_proba(X_test)
				if save_pred:
					save_predictions_csv(clf, X_test, y_test, X_train, y_train, resolution, i, k, pred_path='/raid/users/farzaneh/Histomorphological-Phenotype-Learning/results_250/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/meso_nn250/subtype_clf/clf_csvs/preds/')

				if wandb_tracking:
					wandb_tracking(clf, X_train, y_train, X_test, y_test, labels, resolution, fold_num=i, kernel=k, model_name='SVC')
					print('wandb tracking done for resolution {} and fold {}'.format(resolution, i))
					
	elif mode == 'test':
		i, k = 0, 'rbf'
		data_res_folds = loading_leiden_data(resolutions, h5_complete_path="/raid/users/farzaneh/Histomorphological-Phenotype-Learning/MesoGraph/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_tiles.h5")
		
		for resolution in resolutions:
			# clf_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning/MesoGraph/models/clf_models/SVC_res-{}_fold-{}_kernel-{}.pkl'.format(resolution, i, k)
			# with open(clf_path, 'rb') as file:
				# model = pickle.load(file)
			model = SVC(probability=True, kernel=k, class_weight='balanced')
			data = data_res_folds[resolution][i]['data']
			clf_report(model, data, mode='test')




