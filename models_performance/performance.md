First attemped model: 

    def __init__(
        self,
        n_numeric_features: int,
        categorical_vocab_sizes: list[int],
        categorical_embedding_dims: list[int],
        n_fixture_features: int,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):

fpl_bot.trainer INFO Training complete: 18 epochs, best val loss 3.5198, final val MAE 1.00