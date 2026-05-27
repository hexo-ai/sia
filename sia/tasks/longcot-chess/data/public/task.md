# Chess Hard Task - LLM Evaluation Benchmark

## Task Overview

This task evaluates Large Language Models (LLMs) on 50 challenging chess problems. The questions require the model to analyze chess positions and provide strategic moves using Standard Algebraic Notation (SAN).

### Problem Types

1. **Best 3 Moves**: Given a chess position (FEN notation), find the 3 best next moves
2. **Knight Path**: Calculate minimum moves for a knight to visit all target squares on a 100x100 board

## Dataset Format

### Question Format (`chess_hard.json`)

The dataset contains 50 questions in the following format:

```json
[
  {
    "question_id": "best_3_moves_hard_1",
    "prompt": "You are given a chess position using this FEN: 8/8/4k2K/1pp5/3pP1P1/pP6/P1P5/8 b - - 0 38\nGive the 3 best next moves in this position using the Standard Algebraic Notation (SAN) format. Return your answer in the format: solution = [move1, move2, move3]",
    "problem": {
      "template": "best_3_moves"
    }
  },
  {
    "question_id": "knight_path_hard_1",
    "prompt": "There is a chess board of size 100x100 and certain target squares. Calculate the minimum number of moves it takes for the knight at the given starting position to touch all the target squares...\n\nReturn your answer in the format: solution = <integer>",
    "problem": {
      "template": "knight_path"
    }
  }
]
```

**Important Notes:**
- The dataset file (`chess_hard.json`) does **NOT** contain answers
- Questions use a specific response format that models must follow

### Expected Response Format

Models must respond in one of these formats:

**For best moves (list):**
```
solution = ["c4", "bxc4", "bxc4"]
```

**For knight path (integer):**
```
solution = 218
```

## Result Format for Evaluation

Your inference script must save results in a single `responses.json` file containing all results as an array.

### Responses File Format (`responses.json`)

```json
[
  {
    "question_id": "best_3_moves_hard_1",
    "prompt": "You are given a chess position using this FEN: 8/8/4k2K/1pp5/3pP1P1/pP6/P1P5/8 b - - 0 38\nGive the 3 best next moves...",
    "solution": ["c4", "bxc4", "Kd5"],
    "model_response": "Let me analyze this chess position... solution = [\"c4\", \"bxc4\", \"Kd5\"]",
    "success": true,
    "usage": {
      "prompt_tokens": 123,
      "completion_tokens": 45,
      "total_tokens": 168
    },
    "elapsed_seconds": 1.2,
    "timestamp": "2024-01-15 10:30:45"
  },
  {
    "question_id": "knight_path_hard_1",
    "prompt": "There is a chess board of size 100x100 and certain target squares. Calculate the minimum number of moves...",
    "solution": 218,
    "model_response": "After calculating the optimal knight path... solution = 218",
    "success": true,
    "usage": {
      "prompt_tokens": 125,
      "completion_tokens": 48,
      "total_tokens": 173
    },
    "elapsed_seconds": 1.4,
    "timestamp": "2024-01-15 10:30:46"
  },
  {
    "question_id": "best_3_moves_hard_3",
    "prompt": "You are given a chess position...",
    "solution": null,
    "model_response": "Error occurred during processing",
    "success": false,
    "error": {
      "type": "APIError",
      "message": "Rate limit exceeded"
    },
    "elapsed_seconds": 0.5,
    "timestamp": "2024-01-15 10:30:47"
  }
]
```

### Required Fields for Each Response

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question_id` | string | ✅ **Required** | Must match ID from dataset (e.g., "best_3_moves_hard_1") |
| `prompt` | string | ✅ **Required** | The original question/prompt from dataset |
| `solution` | array/number/null | ✅ **Required** | Extracted answer: array of strings for moves (e.g., `["c4", "bxc4"]`), number for knight path (e.g., `218`), or `null` if failed |
| `model_response` | string | ❌ Optional | Full text response from the model (for debugging) |
| `success` | boolean | ❌ Optional | Whether inference succeeded (default: true if solution is not null) |
| `usage` | object | ❌ Optional | Token usage stats: `{prompt_tokens, completion_tokens, total_tokens}` |
| `elapsed_seconds` | number | ❌ Optional | Time taken for inference |
| `timestamp` | string | ❌ Optional | When inference ran (ISO format or any readable format) |
| `error` | object | ❌ Optional | Error details if failed: `{type, message}` |
| `domain` | string | ❌ Optional | Task domain (e.g., "chess") |
| `difficulty` | string | ❌ Optional | Difficulty level (e.g., "hard") |

### Solution Field Format

The `solution` field must contain the extracted answer in the correct format:

**For best moves questions** (e.g., `best_3_moves_hard_*`):
- Type: `array` of strings
- Format: `["move1", "move2", "move3"]`
- Example: `["c4", "bxc4", "Kd5"]`

**For knight path questions** (e.g., `knight_path_hard_*`):
- Type: `number` (integer)
- Format: Single integer value
- Example: `218`

**For failed inference**:
- Type: `null`
- Use when: Model failed to respond or answer couldn't be extracted

### Directory Structure

Your working directory should look like this:

```
working_dir/
├── responses.json        # All results in a single file (array format)
└── summary.json          # Optional: aggregate stats
```
