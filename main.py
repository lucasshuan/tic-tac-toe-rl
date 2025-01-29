from tic_tac_toe.game import play_tic_tac_toe

def main():
    has_played = False
    while True:
        print("Do you want to play", "another game" if has_played else "a game", "of Tic Tac Toe?")
        answer = input("(y/n): ")
        if answer.lower() != "y":
            print("Thank you for playing!")
            exit()
        
        play_tic_tac_toe()
        has_played = True

if __name__ == "__main__":
    main()