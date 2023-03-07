"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs

import tensorflow as tf
import numpy as np
import logging
import os
from tqdm import tqdm
import argparse

from rmsn.core_routines import train
import rmsn.core_routines as core
import torch as torch
#import rmsn.algorithm as algorithm
import rmsn.algorithm_ood as algorithm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import cloudpickle

#os.environ["CUDA_VISIBLE_DEVICES"]='0'
ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
#MODEL_ROOT = configs.MODEL_ROOT
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)



# EDIT ME! ################################################################################################
# Defines specific parameters to train for - skips hyperparamter optimisation if so
specifications = {
     'rnn_propensity_weighted': (0.1, 4, 100, 64, 0.01, 0.5),
     'treatment_rnn_action_inputs_only': (0.1, 3, 100, 128, 0.01, 2.0),
     'treatment_rnn': (0.1, 4, 100, 128, 0.01, 1.0),
}
####################################################################################################################


def rnn_fit(dataset_map, networks_to_train, MODEL_ROOT, b_use_predicted_confounders,
            b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    

    # Get the correct networks to train
    if networks_to_train == "propensity_networks":
        logging.info("Training propensity networks")
        net_names = ['treatment_rnn_action_inputs_only', 'treatment_rnn']

    elif networks_to_train == "encoder":
        logging.info("Training R-MSN encoder")
        net_names = ["rnn_propensity_weighted"]

    elif networks_to_train == "user_defined":
        logging.info("Training user defined network")
        raise NotImplementedError("Specify network to use!")

    else:
        raise ValueError("Unrecognised network type")

    logging.info("Running hyperparameter optimisation")

    # Experiment name
    expt_name = "treatment_effects"

    # Possible networks to use along with their activation functions
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid')
                      }

    # Setup tensorflow
    tf_device = 'gpu'
    if tf_device == "cpu":
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        # config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        # config.gpu_options.allow_growth = True

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

    training_data = dataset_map['training_data']
    validation_data = dataset_map['validation_data']
    test_data = dataset_map['test_data']

    # Start Running hyperparam opt
    opt_params = {}
    for net_name in net_names:

        print(net_name)

        # Re-run hyperparameter optimisation if parameters are not specified, otherwise train with defined params
        max_hyperparam_runs = 3 if net_name not in specifications else 1

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        use_truncated_bptt = net_name != "rnn_model_bptt" # whether to train with truncated backpropagation through time
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name


       # Extract only relevant trajs and shift data
        training_processed = core.get_processed_data(training_data, b_predict_actions,
                                                     b_use_actions_only, b_use_predicted_confounders,
                                                     b_use_oracle_confounders, b_remove_x1)
        validation_processed = core.get_processed_data(validation_data, b_predict_actions,
                                                       b_use_actions_only, b_use_predicted_confounders,
                                                       b_use_oracle_confounders, b_remove_x1)
        # type(validation_processed['scaled_inputs'])
        # <class 'numpy.ndarray'>
        test_processed = core.get_processed_data(test_data, b_predict_actions,
                                                 b_use_actions_only, b_use_predicted_confounders,
                                                 b_use_oracle_confounders, b_remove_x1)

        num_features = training_processed['scaled_inputs'].shape[-1]
        num_outputs = training_processed['scaled_outputs'].shape[-1]

        # for key in training_processed.keys():
        #     temp = training_processed[key]
        #     if len(temp.shape) >2:
        #         temp = temp.reshape(temp.shape[0],-1)
        #     np.savetxt('/home/defucao/workspace/TSD_pytorch/syc_data/train/'+key+'.csv', temp, delimiter=",")
        # for key in validation_processed.keys():
        #     temp = validation_processed[key]
        #     if len(temp.shape) >2:
        #         temp = temp.reshape(temp.shape[0],-1)
        #     np.savetxt('/home/defucao/workspace/TSD_pytorch/syc_data/val/'+key+'.csv', temp, delimiter=",")
        # for key in test_processed.keys():
        #     temp = test_processed[key]
        #     if len(temp.shape) >2:
        #         temp = temp.reshape(temp.shape[0],-1)
        #     np.savetxt('/home/defucao/workspace/TSD_pytorch/syc_data/test/'+key+'.csv', temp, delimiter=",")
        
        # print('-------finish save----------')



        # Load propensity weights if they exist
        if b_propensity_weight:

            if net_name == 'rnn_propensity_weighted_den_only':
                # use un-stabilised IPTWs generated by propensity networks
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores_den_only.npy"))
            elif net_name == "rnn_propensity_weighted_logistic":
                # Use logistic regression weights
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))
                tmp = np.load(os.path.join(MODEL_ROOT, "propensity_scores_logistic.npy"))
                propensity_weights = tmp[:propensity_weights.shape[0], :, :]
            else:
                # use stabilised IPTWs generated by propensity networks
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))

            logging.info("Net name = {}. Mean-adjusting!".format(net_name))

            propensity_weights /= propensity_weights.mean()

            training_processed['propensity_weights'] = propensity_weights

        # Start hyperparamter optimisation
        hyperparam_count = 0
        while True:

            if net_name not in specifications:

                dropout_rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                memory_multiplier = np.random.choice([0.5, 1, 2, 3, 4])
                num_epochs = 100
                minibatch_size = np.random.choice([64, 128, 256])
                learning_rate = np.random.choice([0.01, 0.005, 0.001])  #([0.01, 0.001, 0.0001])
                max_norm = np.random.choice([0.5, 1.0, 2.0, 4.0])
                hidden_activation, output_activation = activation_map[net_name]

            else:
                spec = specifications[net_name]
                logging.info("Using specifications for {}: {}".format(net_name, spec))
                dropout_rate = spec[0]
                memory_multiplier = spec[1]
                num_epochs = spec[2]
                minibatch_size = spec[3]
                learning_rate = spec[4]
                max_norm = spec[5]
                hidden_activation, output_activation = activation_map[net_name]

            model_folder = os.path.join(MODEL_ROOT, net_name)

            hyperparam_opt = train(net_name, expt_name,
                                  training_processed, validation_processed, test_processed,
                                  dropout_rate, memory_multiplier, num_epochs,
                                  minibatch_size, learning_rate, max_norm,
                                  use_truncated_bptt,
                                  num_features, num_outputs, model_folder,
                                  hidden_activation, output_activation,
                                  config,
                                  "hyperparam opt: {} of {}".format(hyperparam_count,
                                                                    max_hyperparam_runs),
                                   verbose=False)

            hyperparam_count = len(hyperparam_opt.columns)
            # if hyperparam_count >= max_hyperparam_runs:
            #     opt_params[net_name] = hyperparam_opt.T
            #     break
            opt_params[net_name] = hyperparam_opt.T
            break
        logging.info("Done")
        logging.info(hyperparam_opt.T)

        # Flag optimal params
    logging.info(opt_params)


def sc_fit(dataset_map, networks_to_train, MODEL_ROOT, b_use_predicted_confounders,
            b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    

    # Get the correct networks to train
    if networks_to_train == "propensity_networks":
        logging.info("Training propensity networks")
        net_names = ['treatment_rnn_action_inputs_only', 'treatment_rnn']

    elif networks_to_train == "encoder":
        logging.info("Training R-MSN encoder")
        net_names = ["rnn_propensity_weighted"]

    elif networks_to_train == "user_defined":
        logging.info("Training user defined network")
        raise NotImplementedError("Specify network to use!")

    else:
        raise ValueError("Unrecognised network type")

    logging.info("Running hyperparameter optimisation")

    # Experiment name
    expt_name = "treatment_effects"

    # Possible networks to use along with their activation functions
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid')
                      }

    # Setup tensorflow
    tf_device = 'gpu'
    if tf_device == "cpu":
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        # config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        # config.gpu_options.allow_growth = True

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

    training_data = dataset_map['training_data']
    validation_data = dataset_map['validation_data']
    test_data = dataset_map['test_data']

    propensity_score = np.load('/meladyfs/newyork/defucao/NeruIPSCausal/Propensity_score/propensity_scores_spo2.npy')
        
    # zero_arr = np.ones([propensity_score.shape[0],1,1])
    # pre_confounder = np.concatenate((zero_arr,propensity_score), axis = 1)

    training_data['propensity_score'] = propensity_score #pre_confounder

    # Start Running hyperparam opt
    

    # Extract only relevant trajs and shift data
    training_processed = core.get_processed_sc_data(training_data, b_predict_actions = False,
                                                    b_use_actions_only = False, b_use_predicted_confounders=b_use_predicted_confounders,
                                                    b_use_oracle_confounders =False, b_pro_Score=True)
    validation_processed = core.get_processed_sc_data(validation_data, b_predict_actions = False,
                                                    b_use_actions_only = False, b_use_predicted_confounders=b_use_predicted_confounders,
                                                    b_use_oracle_confounders =False, b_pro_Score=False)
    test_processed = core.get_processed_sc_data(test_data, b_predict_actions = False,
                                                    b_use_actions_only = False, b_use_predicted_confounders=b_use_predicted_confounders,
                                                    b_use_oracle_confounders =False, b_pro_Score=False)

    num_features = training_processed['scaled_inputs'].shape[-1]
    num_outputs = training_processed['scaled_outputs'].shape[-1]
    '''
    'outputs': outputs,  # already scaled
        'scaled_inputs': inputs,
        'scaled_outputs': outputs,
        'actions': actions,
        'sequence_lengths': sequence_lengths,
        'active_entries': active_entries
    '''
    #torch.nn.Linear(hidden_channels, 1)

    train_X = torch.from_numpy(training_processed['scaled_inputs']).type(torch.FloatTensor)#[cases,timestapes,freatures]
    # TODO: [cases, timestapes,freatures] --> [cases,timestapes,1]
    len_item = len(training_processed['scaled_outputs'])
    train_y = np.zeros(len_item)

    batch_szie = 16

    for i in range(len_item):
        train_y[i] = training_processed['scaled_outputs'][i][training_processed['sequence_lengths'][i]-2]
    train_y = train_y.reshape([1,len_item,1])
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor)

    len_item = len(test_processed['scaled_outputs'])
    test_y = np.zeros(len_item)

    for i in range(len_item):
        test_y[i] = test_processed['scaled_outputs'][i][test_processed['sequence_lengths'][i]-2]
    test_y = test_y.reshape([1,len_item,1])
    test_y = torch.from_numpy(test_y).type(torch.FloatTensor)
    test_X = torch.from_numpy(test_processed['scaled_inputs']).type(torch.FloatTensor)#[cases,timestapes,freatures]

    train_sequence_length = torch.from_numpy(training_processed['sequence_lengths']).type(torch.IntTensor)
    test_sequence_length = torch.from_numpy(test_processed['sequence_lengths']).type(torch.IntTensor)
    train_active_entries = torch.from_numpy(training_processed['active_entries']).type(torch.IntTensor)
    test_active_entries = torch.from_numpy(test_processed['active_entries']).type(torch.IntTensor)
    train_output = torch.from_numpy(training_processed['scaled_outputs']).type(torch.FloatTensor)
    test_output = torch.from_numpy(test_processed['scaled_outputs']).type(torch.FloatTensor)
    train_pro_score = torch.from_numpy(training_processed['propensity_score']).type(torch.FloatTensor)


    dataset_train_x = torch.from_numpy(training_processed['scaled_inputs']).type(torch.FloatTensor)
    len_item = len(dataset_train_x)
    dataset_train_y = np.zeros(len_item)
    for i in range(len_item):
        dataset_train_y[i] = training_processed['scaled_outputs'][i][training_processed['sequence_lengths'][i]-2]
    dataset_train_y = dataset_train_y.reshape([len(dataset_train_x),1])
    dataset_train_y = torch.from_numpy(dataset_train_y).type(torch.FloatTensor)
    train_dataset = TensorDataset(dataset_train_x, dataset_train_y, train_sequence_length, train_active_entries, train_output,train_pro_score)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_szie, shuffle = False)
    batch_size = 16

    num_left = len(dataset_train_x) - int(len(dataset_train_x)/batch_size)*batch_size # train: 5; test:20
    #For test dataset 
    dataset_test_x = torch.from_numpy(test_processed['scaled_inputs']).type(torch.FloatTensor)
    len_item = len(dataset_test_x)
    dataset_test_y = np.zeros(len_item)
    for i in range(len_item):
        dataset_test_y[i] = test_processed['scaled_outputs'][i][test_processed['sequence_lengths'][i]-2]
    dataset_test_y = dataset_test_y.reshape([len(dataset_test_x),1])
    np.savetxt("./result_label.csv", dataset_test_y, delimiter=",")
    dataset_test_y = torch.from_numpy(dataset_test_y).type(torch.FloatTensor)

    test_dataset = TensorDataset(dataset_test_x, dataset_test_y, test_sequence_length, test_active_entries, test_output)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_szie, shuffle = False)

    #model = algorithm.NeuralCDE(input_channels=training_processed['scaled_inputs'].shape[1], hidden_channels=5)
    device = torch.device("cuda:0") 
    model = algorithm.NeuralCDE(input_channels=training_processed['scaled_inputs'].shape[1],covariates= training_processed['scaled_inputs'].shape[2], hidden_channels=5,device=device).to(device)

    iterations = 10

    


    for epoch in range(10):
        for i,data in enumerate(train_loader):
            inputs, labels, sequence_length, active_entries, output, train_pro_score = data
            #inputs, labels = Variable(inputs), Variable(labels)
            train_X = Variable(inputs).to(device)
            train_y = Variable(labels).unsqueeze(0).to(device)
            sequence_length = Variable(sequence_length).squeeze(0).to(device)
            active_entries = Variable(active_entries).to(device)
            output = Variable(output).to(device)
            train_pro_score = Variable(train_pro_score).to(device)
            #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
            algorithm.train_only(model, train_X, train_y, sequence_length, active_entries, output, iterations, train_pro_score = train_pro_score)
            
            #break

        results = np.zeros((batch_szie,29))
        label = np.zeros((batch_szie,29))
        act = np.zeros((batch_szie,29))
        for i,data  in enumerate(test_loader):
            inputs, labels, sequence_length, active_entries, output = data
            #inputs, labels = Variable(inputs), Variable(labels)

            
            test_X = Variable(inputs).to(device)
            test_y = Variable(labels).unsqueeze(0).to(device)
            sequence_length = Variable(sequence_length).squeeze(0).to(device)
            active_entries = Variable(active_entries)#.to(device)
            output = Variable(output)
            #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
            #algorithm.train_only(model, train_X, train_y, iterations)

            predictions_NC_SC, predictions_output = algorithm.predict(model, test_X, sequence_length)
            
            b, n, s = predictions_output.shape
            b1, n1, s1 = output.shape
            if n != n1:
                continue

            predictions_output = predictions_output.detach().to('cpu').numpy()
            output = output.numpy()
            active_entires = active_entries.numpy()
            results = np.append(results, predictions_output.squeeze(), axis = 0)
            label = np.append(label, output.squeeze(), axis = 0)
            act = np.append(act, active_entires.squeeze(), axis = 0)
            #Y_test_numpy = np.array(test_y).reshape(predictions_NC_SC.shape)
            mses = np.sum(np.sum((predictions_output - output) ** 2 * active_entires, axis=-1), axis=0) \
                   / active_entires.sum(axis=0).sum(axis=-1)

            #print(mses)
            #break
        # print(results.shape)
        np.savetxt("./result_"+str(epoch)+".csv", results[batch_szie:], delimiter=",")
        rmsn_mse = np.sum(np.sum((results[batch_szie:] - label[batch_szie:]) ** 2 * act[batch_szie:], axis=-1), axis=0) \
                    / act[batch_szie:].sum(axis=0).sum(axis=-1)
        rmse = np.sqrt(np.mean(rmsn_mse)) * 100
        print(str(epoch)+":"+str(rmse))

        #np.savetxt("./result_ood_"+str(epoch)+".csv", results[1:], delimiter=",")

        
        with open("./model_ood_"+str(epoch)+'.p', "wb") as f:
            cloudpickle.dump(model, f)
            

    


