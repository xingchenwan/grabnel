import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=2, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.9,0.8,0.7])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    return parser


def update_args_with_default(args):
    """for known datasets, overwrite with the default
     hyperparameters setting provided by the GUNet authors"""

    if args.data == 'COLLAB':
        # see https://github.com/HongyangGao/Graph-U-Nets/blob/master/configs/COLLAB
        args.num_epochs = 200
        args.batch = 64
        args.lr = 0.001
        args.deg_as_tag = 1
        args.l_num = 3
        args.h_dim = 512
        args.l_dim = 64
        args.drop_c = 0.3
        args.drop_c = 0.2
        args.act_c = 'ReLU'
        args.ks = [0.8, 0.6, 0.4]
    elif args.data == 'IMDBMULTI':
        # see https://github.com/HongyangGao/Graph-U-Nets/blob/master/configs/IMDBMULTI
        args.num_epochs = 200
        args.batch = 64
        args.lr = 0.001
        args.deg_as_tag = 1
        args.l_num = 3
        args.h_dim = 512
        args.l_dim = 48
        args.drop_c = 0.1
        args.drop_c = 0.1
        args.act_n = 'LeakyReLU'
        args.act_c = 'ELU'
        args.ks = [0.9, 0.9, 0.9]
    elif args.data == 'PROTEINS':
        # see https://github.com/HongyangGao/Graph-U-Nets/blob/master/configs/PROTEINS
        args.num_epochs = 200
        args.batch = 64
        args.lr = 0.001
        args.deg_as_tag = 0
        args.l_num = 3
        args.h_dim = 512
        args.l_dim = 64
        args.drop_c = 0.3
        args.drop_c = 0.3
        args.act_n = 'ELU'
        args.act_c = 'ELU'
        args.ks = [0.9, 0.8, 0.7]
    elif args.data == 'REDDITMULTI5K':
        # settings not provided by the author, use the same setup as COLLAB since the datasets are of similar scale
        args.num_epochs = 200
        args.batch = 64
        args.lr = 0.001
        args.deg_as_tag = 1
        args.l_num = 3
        args.h_dim = 512
        args.l_dim = 64
        args.drop_c = 0.3
        args.drop_c = 0.2
        args.act_c = 'ReLU'
        args.ks = [0.8, 0.6, 0.4]
    return args

