import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='OTIAS')
    parser.add_argument('--dataroot', type=str, default='E:/HSI_data/cave_x4')
    parser.add_argument('--dataset', type=str, default='cave_x4')
    parser.add_argument('--clip_max_norm', type=int, default=10)
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--model_path', type=str,
                            default='.',
                            help='path for trained encoder')

    # learning settingl
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='end epoch for training')
    parser.add_argument('--n_epochs', type=int, default=1001,
                            help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
 
    args = parser.parse_args()
    return args
 