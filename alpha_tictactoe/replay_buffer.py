import numpy as np

class ReplayBuffer(object):
    """
    Replay buffer that stores a dictionary of data. 
    The dictionary contains data of different labels, where the labels are the key.
    i.e. you can store (x, y) values.

    Usage:
    buffer_config = dict()
    buffer_config["x"] = 4
    buffer_config["y"] = (2, 4)
    replay_buffer = ReplayBuffer(max_entries=10, buffer_config=buffer_config)

    Now the replay buffer will store (x, y) values where x are size 4 vectors
    and y are matrices of shape (2, 4)
    """
    def __init__(self, max_entries, buffer_config):
        """
        Initialize replay buffer that can contain max_entries number of items.
        The data that this can store is specified by buffer_config.

        buffer_config is a dict() that maps from `str` to `int` or `tuple(int)`.
        This means that data with the label specifed by the string will have
        a shape specified by an int or a tuple of ints.
        """
        self.max_entries = max_entries
        
        self.data_dict = dict()
        for key, val in buffer_config.items():
            if (type(val) == int):
                shape = (self.max_entries, val)
            elif (type(val) == tuple):
                shape = (self.max_entries,) + val
            else:
                raise Exception("Not a valid buffer_config.")
            self.data_dict[key] = np.zeros(shape)

        self.start_idx = 0
        self.num_entries = 0

    def add_data(self, data):
        """
        Add data to the replay buffer.
        data is a dictionary with the same keys as buffer_config
        that map to the actual data to add.

        The values can be the same shape specified by buffer config
        to add a single data value.
        Or it can have a batch_size dimension as the first dimension
        to add a batch of data.
        """
        num_data = None
        for key, val in data.items():
            if key not in self.data_dict:
                raise Exception("ReplayBuffer doesn't have key: {}".format(key))
            
            # check if we are extending by a batch of data or just a single
            # data point
            if len(val.shape) == len(self.data_dict[key].shape):
                if num_data is None:
                    num_data = len(val)    
                    assert(num_data < self.max_entries)
                else:
                    assert(num_data == len(val))
                if (self.start_idx + num_data > self.max_entries):
                    split_num_1 = self.max_entries - self.start_idx
                    split_num_2 = num_data - split_num_1
                    self.data_dict[key][self.start_idx:, ...] = val[:split_num_1, ...]
                    self.data_dict[key][:split_num_2, ...] = val[split_num_1:, ...]
                else:
                    self.data_dict[key][self.start_idx:self.start_idx+num_data, ...] = val
            else:
                if num_data is None:
                    num_data = 1
                else:
                    assert(num_data == 1)
                self.data_dict[key][self.start_idx, ...] = val

        self.start_idx += num_data
        if (self.start_idx >= self.max_entries):
            self.start_idx -= self.max_entries
        self.num_entries += num_data
        if self.num_entries > self.max_entries:
            self.num_entries = self.max_entries

    def sample(self, batch_size):
        """
        Sample a batch of data.
        Will return a dictionary that maps from keys specified by
        buffer_config to the sampled data.

        The sampled data will have shape
        (batch_size, shape) where shape is specified by buffer_config.
        """
        if (self.num_entries == 0):
            raise Exception("Can't sample a buffer with no entries.")
        rand_idx = np.random.randint(self.num_entries, size=batch_size)

        sample_data = dict()
        for key, val in self.data_dict.items():
            sample_data[key] = val[rand_idx, ...]
        return sample_data

if __name__ == "__main__":
    
    buffer_config = dict()
    buffer_config["x"] = (2, 4)
    buffer_config["y"] = 5
    buffer = ReplayBuffer(10, buffer_config)


    data_dict = dict()
    data_dict["x"] = np.ones((8, 2, 4))
    data_dict["y"] = np.ones((8, 5))
    buffer.add_data(data_dict)

    print("x:\n{}\n".format(buffer.data_dict["x"]))
    print("y:\n{}\n".format(buffer.data_dict["y"]))

    data_dict["x"] = np.ones((8, 2, 4)) * 2
    data_dict["y"] = np.ones((8, 5)) * 3
    buffer.add_data(data_dict)

    print("x:\n{}\n".format(buffer.data_dict["x"]))
    print("y:\n{}\n".format(buffer.data_dict["y"]))


    sample = buffer.sample(2)
    print(sample)


