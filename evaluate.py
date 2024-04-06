"""
训练并评估单一模型的脚本
"""

import argparse

from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args
import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='MFSTN', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='AISFlow', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=False,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default='None', help='id of experiment') #default=None
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    
    epoch = 36
    exp_id = args.exp_id

    config = ConfigParser(args.task, args.model, args.dataset,
                          args.config_file, args.saved_model, args.train, other_args)
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(args.task), str(args.model), str(args.dataset), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, args.model, args.dataset)
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    
    model_path = executor.cache_dir + '/' + executor.config['model'] + '_' + executor.config['dataset'] + '_epoch%d.tar' % epoch
    checkpoint = torch.load(model_path, map_location='cuda:0')
    executor.model.load_state_dict(checkpoint['model_state_dict'])
    executor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)
