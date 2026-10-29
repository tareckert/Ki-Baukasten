"""
Beispiel 04 – KI-Systeme aus mehreren KIs
=========================================

* Ensemble: Transformer + LSTM + (GRU → MLP)-Pipeline, gelernte Stacking-Kombination
* Unsicherheit: Uneinigkeit der Ensemble-Mitglieder

Ausführen:  python examples/04_kombinationen.py
"""

from ki_baukasten import Project

projekt = Project({
    "name": "zeitreihen_ensemble",
    "data": {"name": "time_series", "args": {"n_samples": 800}},
    "model": {"name": "ensemble", "args": {"combine": "stacking", "members": [
        {"name": "transformer", "args": {"architecture": "encoder_only", "input_type": "continuous", "input_dim": 1,
                                         "output_dim": 4, "pooling": "mean", "d_model": 32, "n_layers": 2,
                                         "positional": "sinusoidal", "max_seq_len": 64}},
        {"name": "rnn", "args": {"input_dim": 1, "output_dim": 4, "hidden_sizes": [32]}},
        {"name": "pipeline", "args": {"stages": [
            {"name": "rnn", "args": {"input_dim": 1, "output_dim": 16, "layer": {"cell": "gru"}}},
            {"name": "mlp", "args": {"output_dim": 4, "hidden_dims": [32]}},
        ]}},
    ]}},
    "training": {"epochs": 8, "optimizer": {"name": "adamw", "args": {"lr": 2e-3}}, "losses": [{"name": "mse"}]},
})
projekt.train()

daten = projekt.data().train["inputs"][:4]
print("Vorhersage:", projekt.predict(daten))
print("Uneinigkeit der Mitglieder (Unsicherheit):", projekt.model.disagreement(daten.to(projekt._device())).mean(-1))
