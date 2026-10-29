"""
Beispiel 07 – Reinforcement Learning mit selbst gestalteten Belohnungen
=======================================================================

* Gitterwelt mit Wänden und Rutschgefahr
* Belohnung = Ziel + Zeitstrafe + Potential-Formung + Wandstrafe + Neuheitsbonus
* EIGENER Belohnungsterm (registriert und sofort per Name nutzbar)
* das Protokoll zeigt den Ertrag je Term (erkennt "Reward Hacking")

Ausführen:  python examples/07_belohnungen_rl.py
"""


from ki_baukasten import Param, Project, T, registry
from ki_baukasten.training.rl import RewardTerm, RewardTermConfig


# --- Eigener Belohnungsterm: Bonus für Bewegung nach rechts ---------------------
class RechtsConfig(RewardTermConfig):
    bonus = Param(0.01, tooltip=T("Bonus je Schritt nach rechts.", "Lenkt die Erkundung.", "Beispiel."))


class RechtsBonus(RewardTerm):
    def forward(self, tr):
        # Beobachtung (coords): [zeile, spalte, ziel_zeile, ziel_spalte] normiert
        return (tr.next_obs[:, 1] > tr.obs[:, 1]).float() * self.cfg.bonus


registry("reward").register("rechts_bonus", lambda a, obs_dim=1: RechtsBonus(a), config=RechtsConfig,
                            tooltip=T("Bonus für Bewegung nach rechts (Beispiel).", "Zeigt eigene Terme.", "Lehrbeispiel."),
                            override=True)

projekt = Project.from_preset("rl_gitterwelt_belohnungen")
belohnungen = list(projekt.get("training.paradigm.rewards"))
belohnungen.append({"name": "rechts_bonus", "args": {"kind": "formend", "anneal": "linear_decay", "anneal_steps": 10000}})
projekt.set("training.paradigm.rewards", belohnungen)
projekt.set("training.paradigm.total_timesteps", 30000)

print(projekt.describe(max_level="advanced", only_changed=True, tooltips=False))
projekt.train()
