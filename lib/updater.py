#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import optimizer as optimizer_module
from chainer import updater
from chainer import Variable
from chainer.dataset import iterator as iterator_module

import copy
import six


class Updater(updater.ParallelUpdater):

    def __init__(self, model, iterator, optimizers, devices):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterator = iterator
        self.iteration = 0

        names = list(six.iterkeys(devices))
        try:
            names.remove('main')
        except ValueError:
            raise KeyError("'devices' must contain a 'main' key.")

        models = {'main': model}
        for name in names:
            m = copy.deepcopy(model)
            if devices[name] >= 0:
                m.to_gpu(devices[name])
            models[name] = m
        if devices['main'] >= 0:
            for optimizer in optimizers:
                optimizer.target.to_gpu(devices['main'])

        self._optimizers = optimizers
        self._devices = devices
        self._models = models

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}
        batch = self.get_iterator('main').next()

        # Split the batch to sub-batches.
        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        # For reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        losses = []
        for model_key, model in six.iteritems(self._models):
            in_arrays = in_arrays_list[model_key]
            loss_func = self.loss_func or model

            if isinstance(in_arrays, tuple):
                in_vars = tuple(Variable(x) for x in in_arrays)
                losses.append(loss_func(*in_vars))
            elif isinstance(in_arrays, dict):
                in_vars = {key: Variable(x)
                           for key, x in six.iteritems(in_arrays)}
                losses.append(loss_func(**in_vars))
            else:
                in_vars = Variable(in_arrays)
                losses.append(loss_func(in_vars))

        # For _uninitialized_params
        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss in losses:
            loss.backward()

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)
