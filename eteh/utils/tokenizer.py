from eteh.reader.txtfile_reader import dict_reader

class Dict(dict):
    def __init__(self, name="", default_key='<unk>', eos_key='<eos>', source={}, filepath=None):
        dict.__init__(source)
        self.name = name
        self.default_key = default_key
        self.eos_key = eos_key
        if filepath is not None:
            self.load_file(filepath)

    def load_file(self, filepath):
        world_dict = dict_reader(filepath, eos=self.eos_key)
        self.update(world_dict)

    def __getitem__(self, i):
        if i in self:
            return self.get(i)
        else:
            return self.get(self.default_key)

class BasicTokenizer(object):
    def encode(self, text):
        token = self.tokenize(text)
        return self.convert_tokens_to_ids(token)

    def tokenize(self, text):
        return [t for t in text]

    def convert_tokens_to_ids(self, tokens):
        return [int(t) for t in tokens]

    def get_dictsize(self):
        return 1

class DictTokenizer(BasicTokenizer):
    def __init__(self, dict_path, eos_key='<eos>', default_key='<unk>', sc=' '):
        self.world_dict = Dict(filepath=dict_path, default_key=default_key, eos_key=eos_key)
        self.eos = eos_key
        self.sc = sc

    def tokenize(self, text):
        if len(self.sc) > 0:
            return text.split(self.sc)
        else:
            return [ch for ch in text]

    def convert_tokens_to_ids(self, tokens):
        return [self.world_dict[t] for t in tokens]

    def get_dictsize(self):
        return len(self.world_dict)