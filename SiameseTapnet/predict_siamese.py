import math
import argparse

from models import TapNet
from Siamese_BipedalWalker.tapnet.read_data import get_data_siamese, get_data_siamese2, get_test_data
from utils import *

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

import Hyperparameter

def load_tapnet_mode():
    parser = argparse.ArgumentParser()
    # cuda settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # Model parameters
    parser.add_argument('--use_cnn', type=boolean_string, default=True,
                        help='whether to use CNN for feature extraction. Default:False')
    parser.add_argument('--use_lstm', type=boolean_string, default=True,
                        help='whether to use LSTM for feature extraction. Default:False')
    parser.add_argument('--use_rp', type=boolean_string, default=True,
                        help='Whether to use random projection')
    parser.add_argument('--rp_params', type=str, default='-1,3',
                        help='Parameters for random projection: number of random projection, '
                             'sub-dimension for each random projection')
    parser.add_argument('--use_metric', action='store_true', default=False,
                        help='whether to use the metric learning for class representation. Default:False')
    parser.add_argument('--filters', type=str, default="256,256,128",
                        help='filters used for convolutional network. Default:256,256,128')
    parser.add_argument('--kernels', type=str, default="8,5,3",
                        help='kernels used for convolutional network. Default:8,5,3')
    parser.add_argument('--dilation', type=int, default=1,
                        help='the dilation used for the first convolutional layer. '
                             'If set to -1, use the automatic number. Default:-1')
    parser.add_argument('--layers', type=str, default="500,300",
                        help='layer settings of mapping function. [Default]: 500,300')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability). Default:0.5')
    parser.add_argument('--lstm_dim', type=int, default=256,
                        help='Dimension of LSTM Embedding.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.sparse = True
    args.layers = [int(l) for l in args.layers.split(",")]
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = Hyperparameter.Step
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = Hyperparameter.Step
        args.rp_params[1] = math.floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(Hyperparameter.Dimension / 64)

    model = TapNet(nfeat=Hyperparameter.Step,
                   len_ts=Hyperparameter.Dimension,
                   layers=args.layers,
                   nclass=Hyperparameter.nclass,
                   dropout=args.dropout,
                   use_lstm=args.use_lstm,
                   use_cnn=args.use_cnn,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   use_metric=args.use_metric,
                   use_rp=args.use_rp,
                   rp_params=args.rp_params,
                   lstm_dim=args.lstm_dim
                   )
    return model

def predict_one(model, seq):
    model.eval()

    bench_noCrash = Hyperparameter.bench_noCrash

    siameseP1 = [bench_noCrash]
    siameseP2 = [seq]

    siameseP1 = torch.FloatTensor(np.array(siameseP1)).cuda()
    siameseP2 = torch.FloatTensor(np.array(siameseP2)).cuda()
    # siameseP3 = torch.FloatTensor(np.array(siameseP3)).cuda()

    output1 = model(siameseP1, siameseP2)
    output1 = torch.nn.Sigmoid()(output1)

    # output2 = model(siameseP3, siameseP2)
    # output2 = torch.nn.Sigmoid()(output2)

    if output1[0][0] > 0.49:
        return 1
    else:
        return 0


if __name__ == "__main__":
    features, labels, idx_train, idx_val, idx_test, nclass = load_features()

    model = load_tapnet_mode()
    model.cuda()
    features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()

    crash_test, noCrash_test = get_test_data(
        features, labels, idx_train, idx_val, idx_test)

    model.load_state_dict(torch.load(r'./data/weights/tapnet.pkl'))

    truth = []  # bench_noCrash
    preds = []

    for i in range(len(crash_test)):
        truth.append(0)
        preds.append(predict_one(model, crash_test[i]))
    for i in range(len(noCrash_test)):
        truth.append(1)
        preds.append(predict_one(model, noCrash_test[i]))

    print(truth)
    print(preds)

    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    f1 = f1_score(truth, preds)

    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)