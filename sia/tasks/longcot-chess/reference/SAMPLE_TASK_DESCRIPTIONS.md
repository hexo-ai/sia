# Sample Task Descriptions

This document contains sample tasks from the longcot evaluation dataset.

---

## Sample Task 1: Chess Best Moves (Hard)

**Question ID:** best_3_moves_hard_1

**Description:**
You are given a chess position using this FEN: `8/8/4k2K/1pp5/3pP1P1/pP6/P1P5/8 b - - 0 38`

Give the 3 best next moves in this position using the Standard Algebraic Notation (SAN) format.

**Return Format:**
```
solution = [move1, move2, move3]
```

**Problem Type:** best_3_moves

---

## Sample Task 2: Knight Path Optimization (Hard)

**Question ID:** knight_path_hard_1

**Description:**
There is a chess board of size 100x100 and certain target squares. Calculate the minimum number of moves it takes for the knight at the given starting position to touch all the target squares. Both target squares and the starting position is given in a 0-indexed format.

**Starting Position:** `(51, 43)`

**Target Squares:**
```
[(64, 27), (9, 77), (53, 72), (26, 16), (18, 97), (46, 3), (84, 17), (66, 45),
 (88, 76), (80, 74), (67, 32), (17, 40), (88, 9), (37, 18), (71, 89), (68, 89),
 (81, 82), (80, 17), (34, 76), (5, 18), (48, 23), (32, 17), (22, 85), (39, 98),
 (9, 9), (19, 93), (80, 9), (50, 92), (70, 38), (1, 74)]
```

**Return Format:**
```
solution = <integer>
```

**Problem Type:** knight_path

---

## Sample Task 3: Chess Best Moves with Complex Position

**Question ID:** best_3_moves_hard_8

**Description:**
You are given a chess position using this FEN: `k3r3/1bR1b1p1/1p2Qn1p/p2p1P2/1q1P1B2/5NP1/6KP/2R5 b - - 3 30`

Give the 3 best next moves in this position using the Standard Algebraic Notation (SAN) format.

**Return Format:**
```
solution = [move1, move2, move3]
```

**Problem Type:** best_3_moves

---

## Sample Task 4: Knight Path with 30 Targets

**Question ID:** knight_path_hard_10

**Description:**
There is a chess board of size 100x100 and certain target squares. Calculate the minimum number of moves it takes for the knight at the given starting position to touch all the target squares. Both target squares and the starting position is given in a 0-indexed format.

**Starting Position:** `(55, 49)`

**Target Squares:**
```
[(34, 64), (49, 24), (68, 89), (47, 25), (29, 57), (48, 97), (48, 15), (47, 18),
 (2, 86), (59, 10), (32, 23), (8, 90), (28, 30), (73, 48), (28, 54), (34, 34),
 (35, 17), (29, 19), (91, 94), (96, 84), (79, 16), (80, 1), (32, 43), (70, 2),
 (99, 26), (55, 54), (24, 34), (11, 82), (58, 28), (35, 70)]
```

**Return Format:**
```
solution = <integer>
```

**Problem Type:** knight_path

---

## Sample Task 5: Chess Tactical Position

**Question ID:** best_3_moves_hard_9

**Description:**
You are given a chess position using this FEN: `r6N/pQpk2pp/3bp3/5b2/7q/1KN1p3/PP1P1PPP/R1B4R b - - 0 14`

Give the 3 best next moves in this position using the Standard Algebraic Notation (SAN) format.

**Return Format:**
```
solution = [move1, move2, move3]
```

**Problem Type:** best_3_moves

---

## Notes

- **Chess Best Moves Tasks:** These require analyzing a chess position given in FEN (Forsyth-Edwards Notation) format and determining the best 3 moves using Standard Algebraic Notation (SAN).

- **Knight Path Tasks:** These are optimization problems requiring calculation of the minimum number of moves for a knight to visit all target squares on a 100x100 chess board. This is a variation of the Traveling Salesman Problem with knight move constraints.

- All tasks are labeled as "hard" difficulty level.
