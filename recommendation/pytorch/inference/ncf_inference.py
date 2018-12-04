import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as mp

import utils
from neumf import NeuMF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)

from mlperf_compliance import mlperf_log

def predict(model, users, items, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    for user, item in batches:
        def proc(x):
            x = np.array(x)
            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda(async=True)
            return torch.autograd.Variable(x)
        outp = model(proc(user), proc(item), sigmoid=True)
        outp = outp.data.cpu().numpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg, len(predictions)


def val_epoch(model, ratings, negs, K, use_cuda=False, output=None, epoch=None,
              processes=1):
    if epoch is None:
        print("Initial evaluation")
    else:
        print("Epoch {} evaluation".format(epoch))

    mlperf_log.ncf_print(key=mlperf_log.EVAL_START, value=epoch)
    start = datetime.now()
    model.eval()
    if processes > 1:
        context = mp.get_context('spawn')
        _eval_one = partial(eval_one, model=model, K=K, use_cuda=use_cuda)
        with context.Pool(processes=processes) as workers:
            hits_ndcg_numpred = workers.starmap(_eval_one, zip(ratings, negs))
        hits, ndcgs, num_preds = zip(*hits_ndcg_numpred)
    else:
        hits, ndcgs, num_preds = [], [], []
        for rating, items in zip(ratings, negs):
            hit, ndcg, num_pred = eval_one(rating, items, model, K, use_cuda=use_cuda)
            hits.append(hit)
            ndcgs.append(ndcg)
            num_preds.append(num_pred)

    hits = np.array(hits, dtype=np.float32)
    ndcgs = np.array(ndcgs, dtype=np.float32)

    assert len(set(num_preds)) == 1
    num_neg = num_preds[0] - 1  # one true positive, many negatives
    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": len(hits) * (1 + num_neg)})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=len(hits))
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=num_neg)

    end = datetime.now()
    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = np.mean(hits)
        result['NDCG'] = np.mean(ndcgs)
        utils.save_result(result, output)

    return hits, ndcgs
use_cuda = torch.cuda.is_available()
model = NeuMF(138493, 26744, mf_dim=64, mf_reg=0., mlp_layer_sizes=[256, 256, 128, 64], mlp_layer_regs=[0. for i in [256, 256, 128, 64]])
model.load_state_dict(torch.load("./trained_models/ncf-model-1543320146.pt", map_location='cpu'))
#print(model)
model.eval()
data = "ml-20m"
topk = 10
processes = 10
test_ratings = load_test_ratings(os.path.join(data, TEST_RATINGS_FILENAME))
test_negs = load_test_negs(os.path.join(data, TEST_NEG_FILENAME))
hits, ndcgs = val_epoch(model, test_ratings, test_negs, topk,
                            use_cuda=use_cuda, processes=1)
print("Hits and ncdgs", hits, ndcgs)