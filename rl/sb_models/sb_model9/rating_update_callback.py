from stable_baselines3.common.callbacks import BaseCallback
from game.enums import Result

class RatingUpdateCallback(BaseCallback):
    def __init__(self, opponent_pool, main_agent_idx=0, verbose=0):
        super().__init__(verbose=verbose)
        self.opponent_pool = opponent_pool
        self.main_agent_idx = main_agent_idx

    def _on_step(self) -> bool:
        # Typically, self.locals["infos"] is a list of 'info' dicts from each parallel env
        infos = self.locals.get("infos", [])
        for env_info in infos:
            # If the environment just finished an episode:
            if "winner" in env_info and env_info["winner"] is not None:
                winner = env_info["winner"]
                opp_idx = env_info.get("opponent_index", None)

                # Convert outcome to Elo result
                if winner == Result.PLAYER_A:
                    result = 1.0
                elif winner == Result.PLAYER_B:
                    result = 0.0
                else:
                    result = 0.5

                if opp_idx is not None:
                    self.opponent_pool.update_rating(self.main_agent_idx, opp_idx, result)
                    if self.verbose:
                        print(f"ELO updated for main={self.main_agent_idx}, opp={opp_idx}, result={result}")

        # Return True so training continues
        return True
