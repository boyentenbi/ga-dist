import pickle
import logging
import redis
import time
import os
import numpy as np
from pprint import pformat
import sys

logger  = logging.getLogger(__name__) # TODO what is __name__ ?

EXP_KEY = "ga:exp"
TASK_ID_KEY = "ga:task_id"
TASK_DATA_KEY = "ga:task_data"
TASK_CHANNEL = 'ga:task_channel'
RESULTS_KEY = 'ga:results'
#NOISES_KEY = 'ga:noises'


def serialize(x):
    return pickle.dumps(x, protocol = -1) # TODO protocol = -1?

def deserialize(x):
    return pickle.loads(x)

def retry_connect(redis_cfg, tries=300, base_delay=4.):
    """
    Connects or tries to reconnect to the server specified by redis_cfg.
    Tries a finite number of times and raises if we run out of tries.

    :param redis_cfg:
    :param tries:
    :param base_delay:
    :return:
    """
    for i in range(tries):
        try:
            r = redis.StrictRedis(**redis_cfg)
            r.ping()
            return r
        except redis.ConnectionError as e:
            if i == tries - 1:
                raise
            else:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                logger.warning('Could not connect to {}. Retrying after {:.2f} sec ({}/{}). Error: {}'.format(
                    redis_cfg, delay, i + 2, tries, e))
                time.sleep(delay)

def retry_get(pipe, key, tries=300, base_delay=4.):
    for i in range(tries):
        # Try to (m)get
        if isinstance(key, (list, tuple)):
            vals = pipe.mget(key)
            if all(v is not None for v in vals):
                return vals
        else:
            val = pipe.get(key)
            if val is not None:
                return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))

class MasterClient:

    """
    One master process in the entire cluster
    """

    def __init__(self, master_redis_cfg):
        # Create the master data store
        self.master_redis = retry_connect(master_redis_cfg)
        self.master_redis.flushdb()
        assert self.master_redis.llen(RESULTS_KEY)==0

        logger.info('[master] connected to Redis: {}'.format(self.master_redis))
        #self.gen_counter = 0

    def declare_experiment(self, exp):
        self.master_redis.set(EXP_KEY, exp)
        logger.info('[master] Declared experiment {}'.format(pformat(exp)))

    # Declare a generation
    def declare_task(self, task_data, gen_num):

        #gen_num = self.gen_counter
        #self.gen_counter += 1

        task_id = np.random.randint(2**31)

        # Serialize the data, ready to send
        gd = serialize(task_data)
        to_publish = serialize((task_id, gd))
        logger.info("Declaring task {} for gen {}. serialized has size {}".format(task_id, gen_num, sys.getsizeof(to_publish)))

        # Create the pipe and send both items at once
        p = self.master_redis.pipeline()
        p.mset({TASK_ID_KEY: task_id,
                TASK_DATA_KEY:gd})
        p.publish(TASK_CHANNEL, to_publish) # TODO serialized serialized?
        p.execute()
        logger.debug('[master] declared task {} for gen {}'.format(task_id, gen_num))
        return task_id

    # Get a result from the relay redis
    def pop_result(self):
        """
        Pops a result from the results
        :return:
        """
        task_id, result = deserialize(self.master_redis.blpop(RESULTS_KEY)[1])
        logger.debug('[master] Popped a result {} for generation with id {}'.format(result, task_id))
        return task_id, result

class RelayClient:
    """
    One relay process per worker NODE
    """

    def __init__(self, master_redis_cfg, relay_redis_cfg):

        # Connect to the existing master redis
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[relay] Connected to master: {}'.format(self.master_redis))

        # Create the relay redis
        self.local_redis = retry_connect(relay_redis_cfg)
        self.local_redis.flushdb()
        assert self.local_redis.llen(RESULTS_KEY)==0
        logger.info('[relay] Connected to relay: {}'.format(self.local_redis))

        self.done = False

    # Continually checks for results in the relay and batches them
    # Pushes every ms to the master (who knows why)
    def run(self):

        # Initialization: read exp and latest gen from master
        self.local_redis.set(EXP_KEY, retry_get(self.master_redis, EXP_KEY))
        task_id, task_data = retry_get(self.master_redis, (TASK_ID_KEY, TASK_DATA_KEY))
        self._declare_task_local(task_id, task_data)

        # Start subscribing to tasks
        p = self.master_redis.pubsub(ignore_subscribe_messages=True)
        p.subscribe(**{TASK_CHANNEL: lambda msg: self._declare_task_local(*deserialize(msg['data']))})
        subscription_thread = p.run_in_thread(sleep_time=0.001)
        batch_sizes = []
        last_print_time = time.time()
        while True:

            # if self.done:
            #     logger.info("[relay] Experiment finished. Closing relay.")
            #     subscription_thread.stop()
            #     p.close()
            #     self.local_redis.flushdb()
            #     break

            results = []
            start_time = curr_time = time.time()
            while curr_time - start_time < 0.001:
                popped = self.local_redis.blpop(RESULTS_KEY, timeout=0)
                results.append(popped[1])
                curr_time = time.time()
            #logger.info("Appending {} results from local redis to master redis".format(len(results)))
            if results:
                self.master_redis.rpush(RESULTS_KEY, *results)

                # Log batch sizes
                batch_sizes.append(len(results))
                if curr_time - last_print_time > 5.0:
                    logger.debug('[relay] Average batch size {:.3f}'.format(sum(batch_sizes) / len(batch_sizes)))
                    last_print_time = curr_time



    def _declare_task_local(self, task_id, task_data):
        logger.info('[relay] Received task {}'.format(task_id))
        self.local_redis.mset({TASK_ID_KEY: task_id, TASK_DATA_KEY: task_data})

        ##self.done = deserialize(task_data).done

    # def relay_noise_lists(self):
    #     self.master_redis.

class WorkerClient:

    """
    One worker thread per core in each worker (except relay core)
    Works concurrently with the actual evaluations
    """

    def __init__(self, relay_redis_cfg):
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[worker] Connected to relay: {}'.format(self.local_redis))
        self.cached_task_id, self.cached_task_data = None, None

    def get_experiment(self):
        exp = deserialize(retry_get(self.local_redis, EXP_KEY))
        logger.info('[worker] Experiment: {}'.format(exp))
        return exp

    def get_current_task(self):

        with self.local_redis.pipeline() as p:
            while True:
                try:
                    # Look for a new gen
                    p.watch(TASK_ID_KEY)
                    task_id = int(retry_get(p, TASK_ID_KEY))
                    if task_id == self.cached_task_id:
                        logger.debug('[worker] Returning cached task_id {}'.format(task_id))
                        break
                    else:
                        # Got a task id without an exception
                        # But its not the same as the cached one
                        p.multi()
                        p.get(TASK_DATA_KEY)
                        logger.info(
                            '[worker] Getting new task with id {}. Cached task was {}'.format(task_id, self.cached_task_id))
                        task_data = p.execute()[0]
                        self.cached_task_data = deserialize(task_data)
                        self.cached_task_id =  task_id

                        break
                except redis.WatchError:
                    # Just try again
                    continue

        return self.cached_task_id, self.cached_task_data

    def push_result(self, task_id, result):
        self.local_redis.rpush(RESULTS_KEY, serialize((task_id, result)))
        logger.debug('[worker] Pushed result for task {}'.format(task_id))

    # def pop_noise_list(self):
    #     task_id, noise_list = deserialize(self.master_redis.blpop(NOISES_KEY)[1])
    #     return task_id, noise_list