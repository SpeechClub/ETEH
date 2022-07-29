class Model_Interface(object):
    def get_input_dict(self):
        raise NotImplementedError("")

    def get_out_dict(self):
        raise NotImplementedError("")

    def train_forward(self, input_dict):
        raise NotImplementedError("")

    def valid_forward(self, input_dict):
        return self.train_forward(input_dict)