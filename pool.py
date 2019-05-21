import torch.multiprocessing as multiprocessing
import time
import os
import psutil
import gc

# Ensure that sharing all the models with all the workers is not too many FDs
multiprocessing.set_sharing_strategy('file_system')

def worker(to_worker, core, all_models):
    psutil.Process().cpu_affinity([core])

    while True:
        f, args = to_worker.get()

        args[0] = all_models[args[0]]
        args[3] = all_models[args[3]]

        f(args)

class Pool(object):
    def __init__(self, cores, maxsize, all_models):
        """ Initialize a pool with cores processes
        """
        self._to_worker = multiprocessing.Queue(maxsize)
        self._processes = []
        self._all_models = all_models

        affinity = list(psutil.Process().cpu_affinity())

        for i in range(cores):
            core = affinity[i % len(affinity)]
            p = multiprocessing.Process(target=worker, args=(self._to_worker, core, all_models), daemon=True)
            p.start()

            self._processes.append(p)

    def map(self, func, args):
        count = len(self._processes)

        # Push functions and arguments to the worker processes
        gc.disable()

        for i, arg in enumerate(args):
            # Replace models by their index
            if func != 'set_all_models':
                arg[0] = self._all_models.index(arg[0])
                arg[3] = self._all_models.index(arg[3])

            self._to_worker.put((func, arg))

        gc.enable()

        return []
