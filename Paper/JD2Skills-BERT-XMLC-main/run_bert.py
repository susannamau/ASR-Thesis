import os
import numpy as np 
import torch
import torch.nn as nn 
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.nn.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport,Accuracy,F1Score    
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

def run_train(args):
    # --------- data ---------
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    idx2word = {}
    for (w,i) in processor.tokenizer.vocab.items():
        idx2word[i] = w

    label_list = processor.get_labels(label_path=config['data_label_path'])                
    idx2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'data_dir'] / f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                                        'data_dir'] / f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_dir'] / "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    # ------- model -------
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list))
    else:
        #STAMPARE QUI I DATI
        print(train_data[:2])
        print(type(train_data))
        model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))

    for p in model.parameters():            
        p.requires_grad=False

    # training last 2 fc layers
    model.classifier.weight.requires_grad = True 
    model.classifier_1.weight.requires_grad = True

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # ---- callbacks ----
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    writer = SummaryWriter()
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(n_gpu=args.n_gpu,i2w=idx2word,i2l=idx2label,
                      model=model,
                      epochs=args.epochs,
                      logger=logger,
                      criterion=BCEWithLogLoss(),
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      early_stopping=None,
                      training_monitor=train_monitor,
                      fp16=args.fp16,
                      resume_path=args.resume_path,
                      grad_clip=args.grad_clip,
                      model_checkpoint=model_checkpoint,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics = [],
                      writer = writer,
                      )
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, seed=args.seed)

def run_test(args):
    from pybert.test.predictor import Predictor
    
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)

    test_data = processor.get_test(config['test_path'])
    test_examples = processor.create_examples(lines=test_data, example_type='test', cached_examples_file=config[
                                                                        'data_dir'] / f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples, max_seq_len=args.eval_max_seq_len, cached_features_file=config[
                                                                        'data_dir'] / "cached_test_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    idx2word = {}
    for (w,i) in processor.tokenizer.vocab.items():
        idx2word[i] = w

    label_list = processor.get_labels(label_path=config['data_label_path'])
    
    idx2label = {i: label for i, label in enumerate(label_list)}
    if args.test_path:
        args.test_path = Path(args.test_path)
        model = BertForMultiLable.from_pretrained(args.test_path, num_labels=len(label_list))
    else:
        model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
    for p in model.bert.parameters():
        p.require_grad = False

    # ----------- predicting -----------
    writer = SummaryWriter()

    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu,
                          i2w = idx2word,
                          i2l = idx2label)
    result = predictor.predict(data=test_dataloader)
    if args.predict_labels:
        predictor.labels(result,args.predict_idx)


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='job_dataset', type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--test_path", default='', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--valid_size", default=0.05, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument('--eval_batch_size', default=4, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=int, )
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=1.0e-4, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--predict_labels', type=bool, default=False)
    parser.add_argument('--predict_idx', type=str, default="0", help=' "idx" or "start-end" or "all" ')

    args = parser.parse_args()
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    init_logger(log_file=config['log_dir'] / f"{args.arch}.log")

    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        from pybert.io.task_data import TaskData
        data = TaskData()
        targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True)
        data.train_val_split(X=sentences, y=targets, shuffle=False, stratify=False,
                             valid_size=args.valid_size, data_dir=config['data_dir'],
                             data_name=args.data_name)
    if args.train:
        run_train(args)

    if args.test:
        run_test(args)


if __name__ == '__main__':
    main()
