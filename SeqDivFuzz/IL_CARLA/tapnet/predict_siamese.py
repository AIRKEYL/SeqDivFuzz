import math
import argparse

from tapnet.models import TapNet
from tapnet.read_data import get_data_siamese, get_data_siamese2, get_test_data
from tapnet.utils import *

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from tapnet import Hyperparameter

def load_tapnet_mode():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.seed = 42
    np.random.seed(42)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.sparse = True
    args.layers = "500,300"
    args.layers = [int(l) for l in args.layers.split(",")]
    args.kernels = "8,5,3"
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = "256,256,128"
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = '-1,3'
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = Hyperparameter.Step
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = Hyperparameter.Step
        args.rp_params[1] = math.floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]

    args.dilation = 1
    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(Hyperparameter.Dimension / 64)

    model = TapNet(nfeat=Hyperparameter.Step,
                   len_ts=Hyperparameter.Dimension,
                   layers=args.layers,
                   nclass=Hyperparameter.nclass,
                   dropout=0,
                   use_lstm=True,
                   use_cnn=True,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   use_metric=False,
                   use_rp=True,
                   rp_params=args.rp_params,
                   lstm_dim=256
                   )
    return model

def predict_one(model, seq):
    model.eval()

    bench_noCrash = Hyperparameter.bench_noCrash
    bench_crash = Hyperparameter.bench_crash

    siameseP1 = [bench_noCrash]
    siameseP2 = [seq]
    siameseP3 = [bench_crash]

    siameseP1 = torch.FloatTensor(np.array(siameseP1)).cuda()
    siameseP2 = torch.FloatTensor(np.array(siameseP2)).cuda()
    # siameseP3 = torch.FloatTensor(np.array(siameseP3)).cuda()

    output1 = model(siameseP1, siameseP2)
    output1 = torch.nn.Sigmoid()(output1)

    # output2 = model(siameseP3, siameseP2)
    # output2 = torch.nn.Sigmoid()(output2)

    if output1[0][0] > 0.48:
        return 1
    else:
        return 0

def predict_once(model, bench_noCrash, seq):
    siameseP2 = [seq]
    siameseP2 = torch.FloatTensor(np.array(siameseP2)).cuda()

    output1 = model(bench_noCrash, siameseP2)
    output1 = torch.nn.Sigmoid()(output1)

    if output1[0][0] > 0.48:
        return 1
    else:
        return 0


def main():
    features, labels, idx_train, idx_val, idx_test, nclass = load_features()

    model = load_tapnet_mode()
    model.cuda()
    features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()

    crash_test, noCrash_test = get_test_data(
        features, labels, idx_train, idx_val, idx_test)

    model.load_state_dict(torch.load(r'./tapnet/data/weights/tapnet.pkl'))

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

if __name__ == "__main__":
    main()