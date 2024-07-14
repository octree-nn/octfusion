
import pickle
from collections import OrderedDict
import os
import ntpath
import time

from termcolor import colored
from . import util

import torch
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.isTrain = opt.isTrain
        self.gif_fps = 4

        if self.isTrain:
            # self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.log_dir = os.path.join(opt.logs_dir, opt.name)

            self.train_img_dir = os.path.join(self.log_dir, 'train_temp')
            self.test_img_dir = os.path.join(self.log_dir, 'test_temp')

        self.name = opt.name
        self.opt = opt

    def setup_io(self):

        if self.isTrain:
            print('[*] create image directory:\n%s...' % os.path.abspath(self.train_img_dir) )
            print('[*] create image directory:\n%s...' % os.path.abspath(self.test_img_dir) )
            util.mkdirs([self.train_img_dir, self.test_img_dir])
            # self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            
            self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
            # with open(self.log_name, "a") as log_file:
            with open(self.log_name, "w") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def print_current_errors(self, current_iters, errors, t):
        # message = '(GPU: %s, epoch: %d, iters: %d, time: %.3f) ' % (self.opt.gpu_ids_str, t)
        # message = f"[{self.opt.exp_time}] (GPU: {self.opt.gpu_ids_str}, iters: {current_iters}, time: {t:.3f}) "
        message = f"[{self.opt.name}] (GPU: {self.opt.gpu_ids_str}, iters: {current_iters}, time: {t:.3f}) "
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(colored(message, 'magenta'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        self.log_tensorboard_errors(errors, current_iters)


    def log_tensorboard_errors(self, errors, cur_step):
        writer = self.opt.writer

        for label, error in errors.items():
            writer.add_scalar('losses/%s' % label, error, cur_step)
