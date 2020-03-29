# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch

from advertorch.utils import get_accuracy
from advertorch.utils import predict_from_logits

from advertorch_examples.models import LeNet5Madry

from advertorch_examples.models import get_cifar10_wrn28_widen_factor
from advertorch_examples.utils import get_test_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default="cuda")
    parser.add_argument('--deterministic', default=False, action="store_true")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--norm', required=True, choices=("Linf", "L2"))
    parser.add_argument('--eps', required=True, type=float)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--test_size', default=None, type=int)


    args = parser.parse_args()

    ckpt = torch.load(args.model)


    if args.dataset.upper() == "CIFAR10":
        model = get_cifar10_wrn28_widen_factor(4)
    elif args.dataset.upper() == "MNIST":
        model = LeNet5Madry()
    else:
        raise

    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    print("model loaded")

    test_loader = get_test_loader(
        args.dataset.upper(), test_size=args.test_size, batch_size=10000)

    import sys
    # TODO: make this more general
    sys.path.append("/home/gavin/Work/ForMSA/auto-attack")
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.eps, plus=True)
    print(adversary.attacks_to_run)

    for data, label in test_loader:
        data, label = data.to(args.device), label.to(args.device)

    pred = predict_from_logits(adversary.model(data))
    adv = adversary.run_standard_evaluation(data, label, bs=2000)
    # 2000 only for MNIST
    advpred = predict_from_logits(adversary.model(adv))

    # adversary.predict = adversary.model
    # adversary.perturb = adversary.run_standard_evaluation

    # adv, label, pred, advpred = attack_whole_dataset(
    #     adversary, test_loader, device=args.device)

    print(get_accuracy(advpred, label))
    print(get_accuracy(advpred, pred))

    # torch.save({"adv": adv}, os.path.join(
    #     os.path.dirname(args.model), "advdata_eps-{}.pt".format(args.eps)))
    # torch.save(
    #     {"label": label, "pred": pred, "advpred": advpred},
    #     os.path.join(os.path.dirname(args.model),
    #                  "advlabel_eps-{}.pt".format(args.eps)))
