from torch import nn


def cnn_categorization_base(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add layers as specified in netspec_opts to the network
    kernel_size, num_filters, stride, layer_type = netspec_opts.get('kernel_size'), netspec_opts.get('num_filters'), \
                                                   netspec_opts.get('stride'), netspec_opts.get('layer_type')

    prev_filter = 3
    conv_l = 1
    bn_l = 1
    relu_l = 1
    for l, l_type in enumerate(layer_type):
        if l_type == 'conv':
            net.add_module(f'conv_{conv_l}',
                           nn.Conv2d(prev_filter, num_filters[l], kernel_size[l], stride[l],
                                     int((kernel_size[l] - 1) / 2)))
            prev_filter = num_filters[l]
            conv_l += 1
        elif l_type == 'bn':
            net.add_module(f'bn_{bn_l}', nn.BatchNorm2d(prev_filter))
            bn_l += 1
        elif l_type == 'relu':
            net.add_module(f'relu_{relu_l}', nn.ReLU())
            relu_l += 1
        elif l_type == 'pool':
            net.add_module(f'pool_{conv_l}', nn.AvgPool2d(kernel_size[l], stride[l], 0))
            conv_l += 1

    return net
