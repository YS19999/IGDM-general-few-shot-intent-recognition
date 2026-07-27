import datetime
import json
import os
import random
import re
import shutil
import sys
import time
import argparse
import logging

import numpy
import numpy as np
from termcolor import colored
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import T5Tokenizer

from dataset import _get_dataloader
from models import T5ModelForIR

import warnings


warnings.filterwarnings('ignore')
similarity = []

def get_label_name(predicts, id2label):

    pred_name = []
    for p in predicts:
        if p not in id2label:
            pred_name.append(p)
        else:
            pred_name.append(id2label[p].lower())
    return pred_name


def dev(model, test_dataloader, tokenizer):

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=object)
    labels_all = np.array([], dtype=object)

    tqdm_test = tqdm(test_dataloader, total=len(test_dataloader), ncols=100, leave=False,
                     desc=colored('Testing', 'yellow'))
    for ind, (inputs, inputs_instruct, label, id2label) in enumerate(tqdm_test):
        inputs["input_ids"] = torch.cat([inputs_instruct["input_ids"][:, :-1], inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.cat([inputs_instruct["attention_mask"][:, :-1], inputs["attention_mask"]], dim=1)

        out = model.generate(inputs)
        label_decode = tokenizer.batch_decode(label['input_ids'], skip_special_tokens=True)
        predict = tokenizer.batch_decode(out, skip_special_tokens=True)
        label_decode = get_label_name(label_decode, id2label)
        predict = get_label_name(predict, id2label)

        labels_all = np.append(labels_all, label_decode)
        predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average="macro")
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, f1, report, total_loss / len(test_dataloader), confusion


def train_batch(args, batch, model, optimizer, scheduler, loss_total, loss_X_total):

    inputs, inputs_instruct, labels, id2label = batch

    X_ebd = model.get_embedding(inputs)
    I_ebd = model.get_embedding(inputs_instruct)
    X_ebd = X_ebd.detach()
    ori_ebd = X_ebd.mean(dim=1)

    # 最大化样本分布
    X_optimizer = torch.optim.SGD([X_ebd.requires_grad_()], lr=args.X_lr)
    for i in range(args.MAX):
        X_optimizer.zero_grad()
        outputs = model.set_forward(X_ebd, inputs, labels)
        X_loss = outputs.loss
        loss_X_total.append(X_loss.item())
        (-X_loss).backward()
        X_optimizer.step()

    X_ebd = X_ebd.detach()
    adv_ebd = X_ebd.mean(dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(ori_ebd, adv_ebd, dim=1)
    similarity.extend(cos_sim.detach().cpu().tolist())

    X_ebd = torch.cat([I_ebd[:, :-1, :], X_ebd], dim=1)
    inputs["attention_mask"] = torch.cat([inputs_instruct["attention_mask"][:, :-1], inputs["attention_mask"]], dim=1)

    # 最小化模型学习目标
    optimizer.zero_grad()
    outputs = model.set_forward(X_ebd, inputs, labels)
    loss = outputs.loss
    loss_total.append(loss.item())
    loss.backward()
    optimizer.step()
    scheduler.step()


def train(args, model, train_dataloader, valid_dataloader, tokenizer):

    best_path = str()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_train_optimization_steps = len(train_dataloader) * args.epoch
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             int(num_train_optimization_steps * args.warmup_proportion),
                                                             num_train_optimization_steps)

    print("{}, Start training".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)
    similarities = []
    loss_total = []
    loss_X_total = []
    patience = 0
    top_acc = 0
    for epoch in range(args.epoch):
        model.train()
        start_time = time.time()

        tqdm_bar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100, leave=False, desc=colored('Training on train', 'yellow'))
        for batch in tqdm_bar:
            train_batch(args, batch, model, optimizer, scheduler, loss_total, loss_X_total)
        similarities.append(np.mean(similarity))
        with torch.no_grad():
            acc, f1, report, loss, confusion = dev(model, valid_dataloader, tokenizer)

        print("{}, Epoch: {:1d}, {:s}{:>7.4f}, {:s}{:>7.4f}, loss:={:>7.4f}  X_loss:={:>7.4f}  cost time: {:>7.4f}".format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
            epoch,
            colored("acc:"), acc,
            colored("f1:"), f1,
            np.mean(np.array(loss_total)),
            np.mean(np.array(loss_X_total)),
            time.time() - start_time
        ), flush=True)

        if top_acc < acc:
            parent_dir = os.path.dirname(best_path)
            if os.path.exists(parent_dir):
                shutil.rmtree(parent_dir)
            top_acc = acc
            best_path = os.path.join(args.save_path, f"{args.dataset}/")
            if not os.path.exists(best_path):
                os.makedirs(best_path)

            best_path = os.path.join(best_path, str(epoch))
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                best_path + ".pth"))

            model.model.save_pretrained("./OUR-T5-IR")

            torch.save(model.state_dict(), best_path + ".pth")
            patience = 0
        else:
            patience += 1

        if patience == args.patience:
            break
    with open("similar.json", "w", encoding="utf-8") as f:
        json.dump(similarities, f)
    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')),
        flush=True)
    model.load_state_dict(torch.load(best_path + '.pth'))
    parent_dir = os.path.dirname(best_path)
    shutil.rmtree(parent_dir)
    # model.model.save_pretrained("./OUR-T5-IR")


def set_seed(seed):

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parsers():
    parser = argparse.ArgumentParser(description='intent recognition with multi-task meta-regularization')

    parser.add_argument("--dataset", type=str, default="meta-train",
                        help="训练")
    parser.add_argument("--shot", type=int, default=3)

    parser.add_argument("--MAX", type=int, default=5)
    parser.add_argument("--X_lr", type=float, default=80)
    parser.add_argument("--meta_train", type=bool, default=True)

    parser.add_argument("--save_path", type=str, default="best-runs/")
    parser.add_argument("--result_path", type=str, default="results_train.jsonl")
    parser.add_argument("--pretrained_path", type=str, default="../flan-t5",
                        help="pre-train model path")

    parser.add_argument("--seed", type=int, default=330)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--test_batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def main():
    args = parsers()

    set_seed(args.seed)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d:%(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_filename = f"{args.dataset}.log"
    if not os.path.exists("log"):
        os.makedirs("log")
    logger.addHandler(logging.FileHandler(os.path.join("log", log_filename), 'w'))
    logger.info(args)

    print("{}, Start training {}-shot model".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), args.shot),
          flush=True)

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    train_dataloader, valid_dataloader, test_dataloader = _get_dataloader(args, tokenizer, args.meta_train)

    model = T5ModelForIR(args)

    if torch.cuda.is_available():
        model = model.cuda(args.cuda)
        # Meta = torch.nn.DataParallel(Meta)

    if args.mode == "train":
        train(args, model, train_dataloader, test_dataloader, tokenizer)

    acc, f1, report, loss, confusion = dev(model, test_dataloader, tokenizer)

    print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
        colored("test acc:", "blue"),
        np.mean(acc),
        colored("test f1:", "blue"),
        np.mean(f1),
    ), flush=True)

    logger.info("Final test accuracy: %.4f" % acc + "\t Final test f1-score: %.4f" % f1)

    with open(f"log/{args.dataset}-confusion.txt", "w", encoding="utf-8") as f:
        f.writelines(str(confusion.tolist()))

    if args.result_path:

        result = {
            "test_acc": acc,
            "test_f1": f1,
        }

        name = ['dataset', 'lr', 'MAX', 'X_lr', 'seed', 'data_path', 'pretrained_path']
        for attr, value in sorted(args.__dict__.items()):
            if attr in name:
                result[attr] = value

        with open(args.result_path, "a", encoding='UTF-8') as f:
            result = json.dumps(result, ensure_ascii=False)
            f.writelines(result + '\n')


if __name__ == "__main__":
    main()
