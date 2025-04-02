

def main():
    import os    
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "workspace"))

    parser = argparse.ArgumentParser(description='Run a game between two players')
    parser.add_argument('--a_name', '-a', type=str, help='Name of player A submission', required=True)
    parser.add_argument('--b_name', '-b', type=str, help='Name of player B submission', required=True)
    parser.add_argument('--map_name', '-m', type=str, help='Name of map to play', required=True)
    
    args = parser.parse_args(sys.argv[1:])
    a_name = args.a_name
    b_name = args.b_name
    map_name = args.map_name

    # Now that we actually have a_name/b_name, we can build a_sub/b_sub:
    submission_dir = os.path.join(os.getcwd(), "workspace") 
    a_sub = os.path.join(submission_dir, a_name)
    b_sub = os.path.join(submission_dir, b_name)

    run_match(a_sub, b_sub, a_name, b_name, map_name)

def run_match(a_sub, b_sub, a_name, b_name, map_name):

    import os

    from gameplay import play_game
    from game.enums import Result

    import time
    import json
    

    map_json_dir = os.path.join(os.getcwd(), "maps.json")

    with open(map_json_dir) as json_file:
        map_dict = json.load(json_file)
        map_string = map_dict[map_name]
    
    if(map_string is None):
        print("map not found")
        return

    for _ in range(1):
        sim_time = time.perf_counter()
        final_board = play_game(map_string, a_sub, b_sub, a_name, b_name, display_game=False, clear_screen = True, record=True, limit_resources=False)


        if (final_board.get_winner()== Result.PLAYER_A):
            print("a won by "+final_board.get_win_reason())
        elif(final_board.get_winner() == Result.PLAYER_B):
            print("b won by "+final_board.get_win_reason())
        elif(final_board.get_winner() == Result.TIE):
            print("tie by "+final_board.get_win_reason())

        sim_time = time.perf_counter() - sim_time
        turn_count = final_board.turn_count
        print(f"{sim_time} seconds elapsed for {turn_count} rounds.")

        


        out_dir = os.path.join(os.getcwd(), "game_env", "match_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_file = 'result.json'
        with open(os.path.join(out_dir, out_file), 'w') as fp:
            fp.write(final_board.get_history_json())


if __name__=="__main__":
    main()
