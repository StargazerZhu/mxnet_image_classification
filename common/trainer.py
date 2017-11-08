from mxnet import gluon, nd
from mxnet import optimizer as opt
from mxnet.model import _create_kvstore
import logging

class RevisedTrainer(gluon.Trainer):
    '''Revise some observed bugs of gluon.Trainer in distributed training
    
    When I was doing distributed training over multiple machines, I found problems to initialize optimizer.
    
    More specifically, when the kvstore sets up the optimizer, there is a EOFError from the server side.
    It seems that initializing optimizer with Parameter dictionary has some problem and I try to initialize it
     with existing idx2name dict, and it works now.
    
    Another weird thing for gluon is that updating on kvstore is disabled for distributed training. I revised
    these codes by following `init_optimizer` in mx.module.Module.
    
    The last revision is the input parameter. The batch size should be initialized for trainer, but it can be changed
    during training, although not tested if this is correct.
    
    '''
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device', batch_size=128):
        super(RevisedTrainer, self).__init__(params, optimizer, optimizer_params, kvstore)
        self._batch_size = batch_size

    def _init_optimizer(self, optimizer, optimizer_params):
        idx2name = {i: param.name for i, param in enumerate(self._params)}
        if isinstance(optimizer, opt.Optimizer):
            assert not optimizer_params, \
                "optimizer_params must be None if optimizer is an instance of " \
                "Optimizer instead of str"
            self._optimizer = optimizer
            self._optimizer.idx2name = idx2name
        else:
            self._optimizer = opt.create(optimizer, param_idx2name=idx2name,
                                         **optimizer_params)

        self._updaters = [opt.get_updater(self._optimizer) \
                          for _ in self._contexts]

    def _init_kvstore(self):
        arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
        kvstore, update_on_kvstore = _create_kvstore(self._kvstore, len(self._contexts),
                                                     arg_arrays)

        if kvstore:
            for i, param in enumerate(self._params):
                param_arrays = param.list_data()
                # logging.debug('Initializaing kvstore for ' + param.name)
                kvstore.init(i, param_arrays[0])
                if update_on_kvstore:
                    # logging.debug('Pulling from kvstore for ' + param.name)
                    kvstore.pull(i, param_arrays, priority=-i)

        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            self._batch_size *= kvstore.num_workers
        self._optimizer.rescale_grad = self._scale / self._batch_size

        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore

        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)

        self._kv_initialized = True
