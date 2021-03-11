import collections

config = collections.namedtuple("config", [
    "cnn_channels", "rnn_hidden_size",
    "lr", "rank", "islstm", "isBi","inp_size",
])

def RC01():
    return config(cnn_channels=600,
                  rnn_hidden_size=300,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True)

def RC02():
    return config(cnn_channels=400,
                  rnn_hidden_size=200,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True,
                  inp_size=32)

def RC03():
    return config(cnn_channels=400,
                  rnn_hidden_size=200,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True,
                  inp_size=64)
