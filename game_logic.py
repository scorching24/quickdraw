import random

VALID_MOVES = ["reload", "shoot", "sheath", "slash", "shield", "deflect"]

def random_ai_move():
    return random.choice(VALID_MOVES)

def run_match(best_of=5):
    target_wins = best_of // 2 + 1
    p1 = PlayerState("you")
    p2 = PlayerState("ai")
    round_num = 1

    print(f"best of {best_of}")
    
    while True:
        p1_move = input(f"round {round_num}")
        print(f"you - gun: {p1.gun_loaded} sword: {p1.sword_sheathed}")
        print(f"ai - gun: {p2.gun_loaded} sword: {p2.sword_sheathed}")

        if p1_move not in VALID_MOVES:
            print("not a valid move")
            continue

        p2_move = random_ai_move()
        result = resolve_round(p1, p1_move, p2, p2_move)

        print(f"you {p1_move:8} | ai: {p2_move:8} -> {result}")

        if result == "match_tie":
            print("duel ends in a draw")
            break

        if result != "continue":
            print(f"score - you: {p1.round_wins} ai: {p2.round_wins}")

        if p2.round_wins >= target_wins:
            print("ai wins")
            break
        round_num += 1
        print()

class PlayerState:
    def __init__(self, name):
        self.name = name
        self.gun_loaded = False
        self.sword_sheathed = False
        self.round_wins = 0

    def apply_move(self, move):
        if move == "reload":
            self.gun_loaded = True
        elif move == "sheath":
            self.sword_sheathed = True
        elif move == "shoot":
            self.gun_loaded = False
        elif move == "slash":
            self.sword_sheathed = False
        

def attack_type_for(move, state):
    if move == "shoot" and state.gun_loaded:
        return "shoot"
    if move == "slash" and state.sword_sheathed:
        return "slash"
    return None

def is_blocked(attack_type, defender_move):
    if defender_move == "shield":
        return True
    if defender_move == "deflect" and attack_type == "shoot":
        return True
    return False

def resolve_round(p1, p1_move, p2, p2_move):
    p1_attack = attack_type_for(p1_move, p1)
    p2_attack = attack_type_for(p2_move, p2)

    p1.apply_move(p1_move)
    p2.apply_move(p2_move)

    if p1_attack is not None and p1_attack == p2_attack:
        p1.gun_loaded = False
        p1.sword_sheathed = False
        p2.gun_loaded = False
        p2.sword_sheathed = False
        return "round_tie"
    
    p1_hits = p1_attack is not None and not is_blocked(p1_attack, p2_move)
    p2_hits = p2_attack is not None and not is_blocked(p2_attack, p1_move)

    if p1_attack == "shoot" and p2_move == "deflect":
        p2.round_wins += 1
        return p2.name
    if p2_attack =="shoot" and p1_move == "deflect":
        p1.round_wins += 1
        return p1.name

    if p1_hits and p2_hits:
        if p1_attack == "shoot":
            p1.round_wins += 1
            return p1.name
        else:
            p2.round_wins += 1
            return p2.name
    elif p1_hits:
        p1.round_wins += 1
        return p1.name
    elif p2_hits:
        p2.round_wins += 1
        return p2.name
    else:
        return "continue"
    
if __name__ == "__main__":
    run_match()