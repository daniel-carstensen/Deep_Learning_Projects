from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE

# Specify the lead_data arguments
# data_path
# mean_subtraction
# normalization

xor_dataset = load_dataset(dataset_path='xor_dataset.pt', mean_subtraction=False, normalization=False)

# specify the network architecture
# in_features
input_features = 2
# out_size
output_size = 2
# hidden_units
hidden_units = [3]
# non_linearity
non_linearity = ['tanH']

# create a network base on the architecture
# net
net = create_net(input_features, hidden_units, non_linearity, output_size)

# specify the training opts
# train_opts
train_opts = {'num_epochs': 25,
              'lr': 0.5,
              'momentum': 0.9,
              'weight_decay': 0,
              'batch_size': 4,
              'step_size': 25,
              'gamma': 1}

# train  and save the model
train(net, xor_dataset, train_opts)
save(net, 'xor_solution.pt')
