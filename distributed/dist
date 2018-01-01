import pickle
import logging
import redis
import time
import os
logger  = logging.getLogger(__name__) # TODO what is __name__ ?


TASK_ID_KEY = "ga:task_id"
TASK_DATA_KEY = "ga:task_data"
TASK_CHANNEL = 'ga:task_channel'
RESULTS_KEY = 'ga:results'


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
    def __init(self, master_redis_cfg):
        self.task_counter = 0
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[master] connected to Redis: {}'.format(self.master_redis))

    def declare_task(self, exp, task_data):
        task_id = self.task_counter
        self.task_counter += 1

        # Serialize the data, ready to send
        serialized_task_data = serialize(task_data)

        # Create the pipe and send both items at once
        pipe = self.master_redis.pipeline()
        pipe.mset({TASK_ID_KEY: task_id, TASK_DATA_KEY:serialized_task_data})
        pipe.publish(TASK_CHANNEL, serialize((task_id, serialized_task_data))) # TODO serialized serialized?
        logger.debug('[master] declared task {}'.format(task_id))
        return task_id


    def pop_result(self):
        """
        Pops a result from the results
        :return:
        """
        task_id, result = deserialize(self.master_redis.blpop(RESULTS_KEY)[1])
        logger.debug('[master] Popped a result for task {}'.format(task_id))
        return task_id, result


