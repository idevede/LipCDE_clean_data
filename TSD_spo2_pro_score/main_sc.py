import os
import argparse
import pickle
import logging

import numpy as np

from simulated_autoregressive import AutoregressiveSimulation
from time_series_deconfounder import time_series_deconfounder
from utils.evaluation_utils import load_results
from sklearn.model_selection import ShuffleSplit
from time_series_deconfounder import get_dataset_splits, train_rmsn, train_sc

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.6, type=float)
    parser.add_argument("--num_simulated_hidden_confounders", default=1, type=int)
    parser.add_argument("--num_substitute_hidden_confounders", default=5, type=int)
    parser.add_argument("--results_dir", default='results_h_s_5')
    parser.add_argument("--exp_name", default='test_tsd_gamma_0.6')
    parser.add_argument("--b_hyperparm_tuning", default=False)
    parser.add_argument("--train_sc", action='store_false')
    parser.add_argument("--train_and_get_confounder", action='store_true')
    parser.add_argument("--train_rmsn", action='store_true')# use --train_rmsn to turn on this option.
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = init_arg()

    model_name = 'factor_model'
    hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, model_name)

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)


    dataset_with_confounders_filename = '{}/{}_dataset_with_substitute_confounders.txt'.format(args.results_dir,
                                                                                               args.exp_name)
    factor_model_hyperparams_file = '{}/{}_factor_model_best_hyperparams.txt'.format(args.results_dir, args.exp_name)

    # if args.train_and_get_confounder:
    #     np.random.seed(100)
    #     autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)
    #     dataset = autoregressive.generate_dataset(5000, 31)
    #     #接下来为了获得datasetmap

    #     time_series_deconfounder(dataset=dataset, num_substitute_confounders=args.num_substitute_hidden_confounders,
    #                                   exp_name=args.exp_name,
    #                                   dataset_with_confounders_filename=dataset_with_confounders_filename,
    #                                   factor_model_hyperparams_file=factor_model_hyperparams_file,
    #                                   b_hyperparm_tuning=args.b_hyperparm_tuning)
    if args.train_sc:
        np.random.seed(100)
        # autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)
        # dataset = autoregressive.generate_dataset(5000, 31)
        # #接下来为了获得datasetmap

        # time_series_deconfounder(dataset=dataset, num_substitute_confounders=args.num_substitute_hidden_confounders,
        #                               exp_name=args.exp_name,
        #                               dataset_with_confounders_filename=dataset_with_confounders_filename,
        #                               factor_model_hyperparams_file=factor_model_hyperparams_file,
        #                               b_hyperparm_tuning=args.b_hyperparm_tuning)
        #dataset = load_results(dataset_with_confounders_filename)
        #dataset = load_results('/home/defucao/workspace/TSD_pytorch_rnn_baseline_to_improve_MIMIC/mimic_data/1228/dataset_mimic_30_10_meanbp_raw.txt')
        np.random.seed(100)
        autoregressive = AutoregressiveSimulation(args.gamma, args.num_simulated_hidden_confounders)
        dataset = autoregressive.generate_dataset(5000, 31)
        logging.info('Fitting counfounded recurrent marginal structural networks.')
        shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
        train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
        shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
        train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
        dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)


        logging.info('Fitting counfounded recurrent marginal structural networks.')
        train_sc(dataset_map, 'rmsn_' + str(args.exp_name), b_use_predicted_confounders=True)
        

    # if args.train_rmsn:
    #     dataset = load_results(dataset_with_confounders_filename)
    #     logging.info('Fitting counfounded recurrent marginal structural networks.')
    #     shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    #     train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
    #     shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
    #     train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
    #     dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)


    #     logging.info('Fitting counfounded recurrent marginal structural networks.')
    #     rmse_without_confounders = train_sc(dataset_map, 'rmsn_' + str(args.exp_name), b_use_predicted_confounders=True)
    #     print('********************')
    #     print(rmse_without_confounders)
    #     print('********************')
    #     # logging.info(
    #     #     'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(
    #     #         args.num_substitute_confounders))
    #     # rmse_with_confounders = train_rmsn(dataset_map, 'rmsn_' + str(args.exp_name), b_use_predicted_confounders=True)

    #     print("Outcome model RMSE when trained WITHOUT the hidden confounders.")
    #     print(rmse_without_confounders)

    #     # print("Outcome model RMSE when trained WITH the substitutes for the hidden confounders.")
    #     # print(rmse_with_confounders)
    #     # print('done')
