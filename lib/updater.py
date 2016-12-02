#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import training
from chainer import Variable
from chainer.dataset import iterator as iterator_module
from chainer.dataset import convert
from chainer.training import updater

import chainer
import copy
import logging
import six


class Updater(updater.ParallelUpdater):

    def __init__(self, model, iterator, devices, n_encdec,
                 converter=convert.concat_examples):
        assert isinstance(iterator, iterator_module.Iterator)
        self._iterators = {'main': iterator}
        self.iteration = 0

        names = list(six.iterkeys(devices))
        models = {}
        for name in names:
            m = copy.deepcopy(model)
            if devices[name] >= 0:
                m.to_gpu(devices[name])
            models[name] = m

        self.converter = converter
        self._devices = devices
        self._models = models
        self._optimizers = model.predictor.optimizers
        self._depth = n_encdec

        logging.info('Optimizers in Updater:')
        for name, opt in self._optimizers.items():
            logging.info('{}: {}, {}'.format(name, opt, opt.target.name))

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, d):
        self._depth = d

    def connect_trainer(self, trainer):
        for name, model in self._models.items():
            trainer.reporter.add_observer(name, model)

    def update_core(self):
        model_main = self._models['main']
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        batch = self.get_iterator('main').next()

        # Split the batch to sub-batches for each GPU
        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        # For reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        losses = []
        optimizers = []
        for device_key, model in six.iteritems(self._models):
            in_arrays = in_arrays_list[device_key]
            loss_func = model

            if isinstance(in_arrays, tuple):
                in_vars = [Variable(x) for x in in_arrays] + [self._depth]
                loss, opts = loss_func(*in_vars)
                losses.append(loss)
                for i, opt in enumerate(opts.values()):
                    if self._devices[device_key] >= 0:
                        opt.target.to_gpu(self._devices[device_key])
                    optimizers.append(opt)
            else:
                raise ValueError('Dataset should return a tuple.')

        # For _uninitialized_params
        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss in losses:
            loss.backward()

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        for optimizer in optimizers:
            optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)
