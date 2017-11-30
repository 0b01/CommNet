import numpy as np

NLEVERS = 100

class Levers(object):
    def __init__(self):
        pass

    def step(self, actions):
        rewards = self.__get_rewards(actions)
        # there is no next step
        return rewards

    def __get_rewards(self, actions):
        """
        reward is the ratio of distinct / all
        """
        # XXX: broken
        distinct_num = np.sum(np.cast(np.sum(actions, axis=1) > 0, tf.float32),
                                axis=1, keep_dims=True)
        return distinct_num / NLEVERS
