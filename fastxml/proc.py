from builtins import object
import multiprocessing

class Result(object):

    def ready(self):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

class ForkResult(Result):
    def __init__(self, queue, p):
        self.queue = queue
        self.p = p 

    def ready(self):
        return self.p.is_alive()

    def get(self):
        result = self.queue.get()
        self.p.join()
        self.queue.close()
        return result

class SingleResult(Result):
    def __init__(self, res):
        self.res = res

    def ready(self):
        return True

    def get(self):
        return self.res

def _remote_call(q, f, args):
    results = f(*args)
    q.put(results)

def faux_fork_call(f):
    def f2(*args):
        return SingleResult(f(*args))

    return f2

def fork_call(f):
    def f2(*args):
        queue = multiprocessing.Queue(1)
        p = multiprocessing.Process(target=_remote_call, args=(queue, f, args))
        p.start()
        return ForkResult(queue, p)

    return f2

