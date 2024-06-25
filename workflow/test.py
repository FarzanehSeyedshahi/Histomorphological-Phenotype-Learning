import h5py
import numpy as np
from lifelines import CoxTimeVaryingFitter
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

# # print the h5_temp
main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning/results/BarlowTwins_3'
# h5_temp = h5py.File('/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_test.h5', 'r')
# print('h5_temp', h5_temp.keys())
# for key in h5_temp.keys():
#     print(key)
#     print(h5_temp[key].shape)
#     print(h5_temp[key][0])
#     print(h5_temp[key][-1])
#     print()
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

def train_cox(datas, penalizer, l1_ratio, event_ind_field='event_ind', event_data_field='event_data', robust=True, frame_clusters=None, groupby=None):
	# Train Cox Proportional Hazard.
	train, set_name = datas[0]
	#drop nans
	# train = train.dropna(axis=0, how='any')
    # cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

	cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
	cph.fit(train, duration_col=event_data_field, event_col=event_ind_field, show_progress=False, robust=robust)

	# Partial hazard prediction for each list.
	predictions = list()
	for data, set_name in datas:
		if data is not None:
			pred = cph.predict_partial_hazard(data)
			print(cph.print_summary())
		else:
			pred = None
		predictions.append((pred, set_name))

	summary_table = cph.summary
	if frame_clusters is not None:
		frame_clusters = frame_clusters.sort_values(by=groupby)
		for column in ['coef', 'coef lower 95%', 'coef upper 95%', 'p']:
			for cluster_id in [col for col in train if col not in [event_ind_field, event_data_field]]:
				frame_clusters.loc[frame_clusters[groupby]==int(cluster_id), column] = summary_table.loc[cluster_id, column].astype(np.float32)
		frame_clusters = frame_clusters.sort_values(by='coef')

	return cph, predictions, frame_clusters

# Evaluate survival model: C-Index.
def evalutaion_survival(datas, predictions, event_ind_field='event_ind', event_data_field='event_data', c_index_type='Harrels'):
	cis = list()
	for i, data_i in enumerate(datas):
		data, set_named = data_i
		if data is not None:
			prediction, set_namep = predictions[i]
			# Concordance index for right-censored data.
			if c_index_type=='Harrels':
				c_index = np.round(concordance_index_censored(data[event_ind_field]==1.0, data[event_data_field], prediction)[0],2)
			# Concordance index for right-censored data based on inverse probability of censoring weights
			elif c_index_type=='ipcw':
				train_data = datas[0][0].copy(deep=True)
				train_data[event_ind_field]  = train_data[event_ind_field].astype(bool)
				train_data[event_data_field] = train_data[event_data_field].astype(float)
				data[event_ind_field]       = data[event_ind_field].astype(bool)
				data[event_data_field]      = data[event_data_field].astype(float)
				c_index = concordance_index_ipcw(survival_train=train_data[[event_ind_field,event_data_field]].to_records(index=False), survival_test=data[[event_ind_field,event_data_field]].to_records(index=False), estimate=prediction)[0]
			if set_namep != set_named:
				print('Mismatch between adata and predictions')
				print('Data set:', set_named, 'Prediction set:', set_namep)
				exit()
		else:
			c_index   = None
			set_namep = data_i[1]
		cis.append((c_index, set_namep))
	return cis

def cox_sksur():
	# Xt, y = 
	cox_model = CoxPHSurvivalAnalysis(alpha=0.1, verbose=1)
	
	return cox_model



from lifelines import CoxPHFitter
import pandas as pd
fold = 0
resolution = 7.0
groupby = 'leiden_{}'.format(resolution).replace('.', 'p')
cis = list()
cis_test = list()
for fold in range(5):
    resolution = 2.0
    groupby = 'leiden_{}'.format(resolution).replace('.', 'p')
    data_train = pd.read_csv('{}/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/meso_overal_survival_nn400/survival_csvs/clr_{}_fold{}_train.csv'.format(main_path, groupby, fold), index_col=0).iloc[:,1:]

    data_test = pd.read_csv('{}/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/meso_overal_survival_nn400/survival_csvs/clr_{}_fold{}_additional.csv'.format(main_path, groupby, fold), index_col=0).iloc[:,1:]
    # cph = CoxPHFitter(penalizer=0.1)
    # cph.fit(data_train,'os_event_data', 'os_event_ind')
    # cph.print_summary()
    datas = [(data_train, 'train'), (data_test, 'test')]
    # X, y = load_breast_cancer()
    # print(X.shape)
    # print(y.shape)
    data, set_name = datas[0]
    X, y = data.drop(['os_event_ind', 'os_event_data'], axis=1), data[['os_event_ind', 'os_event_data']]
    X_test, y_test = datas[1][0].drop(['os_event_ind', 'os_event_data'], axis=1), datas[1][0][['os_event_ind', 'os_event_data']]
    # form y to a list of tuples (event_ind(true/false), event_data)
    y = [tuple(x) for x in y.to_numpy()]
    y = np.array(y, dtype=[('event_ind', '?'), ('event_data', '<f8')])
    y_test = [tuple(x) for x in y_test.to_numpy()]
    y_test = np.array(y_test, dtype=[('event_ind', '?'), ('event_data', '<f8')])

    # check time contains values smaller or equal to zero
    u = y['event_data'] <= 0
    if u.any():
        y['event_data'][u] = 0.1
    # cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
    # cox_lasso.fit(X, y)

    # coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.01, alpha_min_ratio=0.01, max_iter=100))
    # warnings.simplefilter("ignore", UserWarning)
    # from sklearn.exceptions import FitFailedWarning
    # warnings.simplefilter("ignore", FitFailedWarning)
    # coxnet_pipe.fit(X, y)
    # print(coxnet_pipe.named_steps["coxnetsurvivalanalysis"].)
    # estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    # cv = KFold(n_splits=5, shuffle=True, random_state=0)
    # gcv = GridSearchCV(
    #     make_pipeline(StandardScaler(),CoxnetSurvivalAnalysis(l1_ratio=0.01)),
    #     param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    #     cv=cv,
    #     error_score=0.5,
    #     n_jobs=1,
    # ).fit(X, y)

    # cv_results = pd.DataFrame(gcv.cv_results_)



    # alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    # mean = cv_results.mean_test_score
    # std = cv_results.std_test_score
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(figsize=(9, 6))
    # ax.plot(alphas, mean)
    # ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    # ax.set_xscale("log")
    # ax.set_ylabel("concordance index")
    # ax.set_xlabel("alpha")
    # ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    # ax.axhline(0.5, color="grey", linestyle="--")
    # ax.grid(True)
    # plt.savefig('{}/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/meso_overal_survival_nn400/survival_csvs/test.png'.format(main_path))

    def score_survival_model(model, X, y):
        prediction = model.predict(X)
        result = concordance_index_censored(y["event_ind"], y["event_data"], prediction)
        return result[0]

    penalizer = 0.1
    l1_ratio = 0.0
    c_index_type = 'Harrels'
    robust = False
    frame_clusters = None
    groupby = None
    from sksurv.metrics import concordance_index_censored
    from sksurv.svm import FastSurvivalSVM, NaiveSurvivalSVM
    from sklearn.model_selection import ShuffleSplit, GridSearchCV

    estimator = FastSurvivalSVM(max_iter=100, tol=1e-5, random_state=0)
    # estimator = CoxPHSurvivalAnalysis()
    # estimator = CoxnetSurvivalAnalysis(l1_ratio=0.01, alpha_min_ratio=0.01, max_iter=100)
    # estimator = NaiveSurvivalSVM()
    estimator.fit(X, y)
    cis_test.append(estimator.score(X_test, y_test))
    cis.append(estimator.score(X, y))
print(np.mean(cis), np.mean(cis_test))
print(cis, cis_test)
import warnings
param_grid = {"alpha": 2.0 ** np.arange(-12, 13, 2)}
# cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
# gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
warnings.filterwarnings("ignore", category=UserWarning)
# gcv = gcv.fit(X, y)
# print(round(gcv.best_score_, 3), gcv.best_params_)
# estimator, predictions, _ = train_cox(datas, penalizer=penalizer, l1_ratio=l1_ratio, event_ind_field='os_event_ind', event_data_field='os_event_data', robust=robust, frame_clusters=frame_clusters, groupby=groupby)
# print(estimator.score(data_test, scoring_method='concordance_index'))
# print(estimator.score(data_train, scoring_method='concordance_index'))

# print(estimator.score(data_test, scoring_method='log_likelihood'))
# print(estimator.score(data_train, scoring_method='log_likelihood'))
# cis = evalutaion_survival(datas, predictions, event_ind_field='os_event_ind', event_data_field='os_event_data', c_index_type=c_index_type)
# print(cis, 'info:', 'l1_ratio:',l1_ratio, 'penalizer:', penalizer, 'robust:',robust)
# print('params_', cph.params_)
# print('baseline_hazard_', cph.baseline_hazard_)
# print('baseline_cumulative_hazard_', cph.baseline_cumulative_hazard_)
# print('baseline_survival_', cph.baseline_survival_)
# print('confidence_intervals_', cph.confidence_intervals_)
# print('concordance_index_', cph.concordance_index_)
# print('log_likelihood_', cph.log_likelihood_)






