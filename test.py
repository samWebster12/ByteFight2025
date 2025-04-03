import os
import ray
from run_game import run_match

# Config
PLAYER_A = "cnn3"
PLAYER_B = "cnn1"
NUM_MATCHES = 100
MAP_NAME = "pillars"

# Initialize Ray
ray.init(ignore_reinit_error=True, num_cpus=128)

@ray.remote(num_cpus=2)
def run_match_remote(a_name, b_name):
    submission_dir = os.path.join(os.getcwd(), "workspace")
    try:
        _, _, winner = run_match(
            submission_dir,  # directory_a
            submission_dir,  # directory_b
            a_name,
            b_name,
            MAP_NAME
        )
        return winner
    except Exception as e:
        print(f"[ERROR] Match crashed: {e}")
        return "Tie"  # Treat crashes as ties

def main():
    futures = [
        run_match_remote.remote(PLAYER_A, PLAYER_B)
        for _ in range(NUM_MATCHES)
    ]

    results = ray.get(futures)

    wins_a = sum(1 for r in results if r == "A")
    wins_b = sum(1 for r in results if r == "B")
    ties   = sum(1 for r in results if r != "A" and r != "B")

    print("\n=== Match Results ===")
    print(f"{PLAYER_A} wins: {wins_a}")
    print(f"{PLAYER_B} wins: {wins_b}")
    print(f"Ties / errors: {ties}")
    print("=====================")

if __name__ == "__main__":
    main()


