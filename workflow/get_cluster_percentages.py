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