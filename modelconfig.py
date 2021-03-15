import collections

config = collections.namedtuple("config", [
    "cnn_channels", "rnn_hidden_size",
    "lr", "rank", "islstm", "isBi", "eps_factor",
])

def RC01():
    return config(cnn_channels=600,
                  rnn_hidden_size=300,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True)

def RC02():
    return config(cnn_channels=600,
                  rnn_hidden_size=300,
                  lr = 0.001,
                  rank=32,
                  islstm=True,
                  isBi=True)

def RC03():
    return config(cnn_channels=400,
                  rnn_hidden_size=200,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True,
                  eps_factor=0.01)

def RC04():
    return config(cnn_channels=400,
                  rnn_hidden_size=200,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True,
                  eps_factor=0.03)

def RC05():
    return config(cnn_channels=400,
                  rnn_hidden_size=200,
                  lr = 0.001,
                  rank=50,
                  islstm=True,
                  isBi=True,
                  eps_factor=0.05)