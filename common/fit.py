# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import logging
import os
import time
from mxnet.gluon.model_zoo import vision as models
from trainer import *
from mxnet import autograd

def _get_mlp():
  net = gluon.nn.Sequential(prefix='mlp_')
  with net.name_scope():
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dense(10))
    return net

def _get_lr_scheduler(args, kv):
  if 'lr_factor' not in args or args.lr_factor >= 1:
    return (args.lr, None)
  epoch_size = args.num_examples / args.batch_size
  if 'dist' in args.kv_store:
    epoch_size /= kv.num_workers
  begin_epoch = args.load_epoch if args.load_epoch else 0
  step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
  lr = args.lr
  for s in step_epochs:
    if begin_epoch >= s:
      lr *= args.lr_factor
  if lr != args.lr:
    logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

  steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
  return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
  if 'load_epoch' not in args or args.load_epoch is None:
    return (None, None, None)
  assert args.model_prefix is not None
  model_prefix = args.model_prefix
  if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
    model_prefix += "-%d" % (rank)
  sym, arg_params, aux_params = mx.model.load_checkpoint(
    model_prefix, args.load_epoch)
  logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
  return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
  if args.model_prefix is None:
    return None
  dst_dir = os.path.dirname(args.model_prefix)
  if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
  return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
    args.model_prefix, rank))

def add_fit_args(parser):
  """
  parser : argparse.ArgumentParser
  return a parser added with args required by fit
  """
  train = parser.add_argument_group('Training', 'model training')
  train.add_argument('--network', type=str,
             help='the neural network to use')
  train.add_argument('--num-layers', type=int,
             help='number of layers in the neural network, required by some networks such as resnet')
  train.add_argument('--gpus', type=str,
             help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
  train.add_argument('--kv-store', type=str, default='device',
             help='key-value store type')
  train.add_argument('--num-epochs', type=int, default=100,
             help='max num of epochs')
  train.add_argument('--lr', type=float, default=0.1,
             help='initial learning rate')
  train.add_argument('--lr-factor', type=float, default=0.1,
             help='the ratio to reduce lr on each step')
  train.add_argument('--lr-step-epochs', type=str,
             help='the epochs to reduce the lr, e.g. 30,60')
  train.add_argument('--optimizer', type=str, default='sgd',
             help='the optimizer type')
  train.add_argument('--mom', type=float, default=0.9,
             help='momentum for sgd')
  train.add_argument('--wd', type=float, default=0.0001,
             help='weight decay for sgd')
  train.add_argument('--batch-size', type=int, default=128,
             help='the batch size')
  train.add_argument('--disp-batches', type=int, default=20,
             help='show progress for every n batches')
  train.add_argument('--model-prefix', type=str,
             help='model prefix')
  parser.add_argument('--monitor', dest='monitor', type=int, default=0,
            help='log network parameters every N iters if larger than 0')
  train.add_argument('--load-epoch', type=int,
             help='load the model on an epoch using the model-load-prefix')
  train.add_argument('--top-k', type=int, default=0,
             help='report the top-k accuracy. 0 means no report.')
  train.add_argument('--test-io', type=int, default=0,
             help='1 means test reading speed without training')
  train.add_argument('--dtype', type=str, default='float32',
             help='precision: float32 or float16')
  train.add_argument('--mode', type=str, default='hybrid',
                     help='mode in which to train the model. options are symbolic, imperative, hybrid')
  return train

def fit(args, network, data_loader, **kwargs):
  """
  train a model
  args : argparse returns
  network : the symbol definition of the nerual network
  data_loader : function that returns the train and val data iterators
  """
  # kvstore
  kv = mx.kvstore.create(args.kv_store)

  # logging
  head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
  logging.basicConfig(level=logging.DEBUG, format=head)
  logging.info('start with arguments %s', args)

  # data iterators
  (train, val) = data_loader(args, kv)
  if args.test_io:
    tic = time.time()
    for i, batch in enumerate(train):
      for j in batch.data:
        j.wait_to_read()
      if (i+1) % args.disp_batches == 0:
        logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
          i, args.disp_batches*args.batch_size/(time.time()-tic)))
        tic = time.time()

    return

  # devices for training
  # devs = [mx.cpu()] if args.gpus is None or args.gpus is '' else [
  #   mx.gpu(int(i)) for i in args.gpus.split(',')]
  devs = [mx.cpu(0), mx.cpu(1)]

  # learning rate
  lr, lr_scheduler = _get_lr_scheduler(args, kv)

  optimizer_params = {
      'learning_rate': lr,
      'wd' : args.wd,
      'lr_scheduler': lr_scheduler
  }

  # Add 'multi_precision' parameter only for SGD optimizer
  if args.optimizer == 'sgd':
    optimizer_params['multi_precision'] = True

  # Only a limited number of optimizers have 'momentum' property
  has_momentum = {'sgd', 'dcasgd', 'nag'}
  if args.optimizer in has_momentum:
    optimizer_params['momentum'] = args.mom

  monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

  if args.network == 'alexnet':
    # AlexNet will not converge using Xavier
    initializer = mx.init.Normal()
  else:
    initializer = mx.init.Xavier(
      rnd_type='gaussian', factor_type="in", magnitude=2)
  # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),

  # evaluation metrices
  eval_metrics = ['accuracy']
  if args.top_k > 0:
    eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

  # callbacks that run after each batch
  batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
  if 'batch_end_callback' in kwargs:
    cbs = kwargs['batch_end_callback']
    batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

  # run
  if isinstance(network, mx.symbol.Symbol) and args.mode == 'symbolic':
    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
      arg_params = kwargs['arg_params']
      aux_params = kwargs['aux_params']
    else:
      sym, arg_params, aux_params = _load_model(args, kv.rank)
      if sym is not None:
        assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)

    # create model
    model = mx.mod.Module(
      context=devs,
      symbol=network
    )

    model.fit(train,
      begin_epoch        = args.load_epoch if args.load_epoch else 0,
      num_epoch          = args.num_epochs,
      eval_data          = val,
      eval_metric        = eval_metrics,
      kvstore            = kv,
      optimizer          = args.optimizer,
      optimizer_params   = optimizer_params,
      initializer        = initializer,
      arg_params         = arg_params,
      aux_params         = aux_params,
      batch_end_callback = batch_end_callbacks,
      epoch_end_callback = checkpoint,
      allow_missing      = True,
      monitor            = monitor)

  else:
    kwargs = {'ctx': devs, 'pretrained': False, 'classes': args.num_classes}

    if network == 'mlp':
      net = _get_mlp()
    else:
      net = models.get_model(network, **kwargs)

    if args.mode == 'hybrid':
      net.hybridize()
    net.collect_params().initialize(ctx=devs, init=initializer, force_reinit=True)

    if args.optimizer == 'admm':
      optimizer_params['N'] = len(devs)
      optimizer_params['gamma'] = 0.0
      trainer = ADMMTrainer(net.collect_params(), optimizer_params, kvstore=kv, batch_size=args.batch_size)
    else:
      # trainer = RevisedTrainer(net.collect_params(), args.optimizer, optimizer_params, kvstore=kv, batch_size=args.batch_size)
      trainer = gluon.Trainer(net.collect_params(), args.optimizer, optimizer_params, kvstore=kv)
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(args.num_epochs):
      tic = time.time()
      train.reset()
      metric.reset()
      btic = time.time()
      for i, batch in enumerate(train):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=devs, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=devs, batch_axis=0)
        outputs = []
        Ls = []
        with autograd.record():
          # Ls = [loss(net(x), y) for x, y in zip(data, label)]
          for x, y in zip(data, label):
            z = net(x)
            L = loss(z, y)
            # store the loss and do backward after we have done forward
            # on all GPUs for better speed on multiple GPUs.
            Ls.append(L)
            outputs.append(z)
        for L in Ls:
          L.backward()

        trainer.step(batch.data[0].shape[0])
        metric.update(label, outputs)
        if args.disp_batches and (i+1)%args.disp_batches==0:
          # weight = net.collect_params()['mlp_dense0_weight']
          # for dev in devs:
          #   logging.debug(weight.grad(ctx=dev))
          name, acc = metric.get()
          logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
            epoch, i, args.batch_size/(time.time()-btic), name, acc))
        btic = time.time()

      name, acc = metric.get()
      logging.info('[Epoch %d] training: %s=%f' % (epoch, name, acc))
      logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

      # testing
      val_metric = mx.metric.Accuracy()
      val.reset()
      for batch in val:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=devs, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=devs, batch_axis=0)
        outputs = []
        for x in data:
          outputs.append(net(x))
        val_metric.update(label, outputs)
      name, val_acc = val_metric.get()
      logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))

    net.save_params('image-classifier-%s-%d.params'%(args.network, args.num_epochs))
