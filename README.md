# Conversation Analysis Tool

This tool provides analysis for conversation data from [OpenAI](https://openai.com/), including identifying common words, modeling topics, and checking for question rephrasing patterns. It allows you to refine the analysis by filtering out common English words, punctuation, special characters, numbers, and single-letter words.

## Requirements

- Python 3.x
- `scikit-learn`
- `argparse`
- `json`

You can install the necessary Python packages with:

```bash
pip install scikit-learn argparse
```

## Usage

**Basic Usage:**

1. **Common Words Analysis:**
   Identify the most common words in your conversation data.

   ```bash
   python3 parse3.py ./conversations.json --common-words --top-n 30
   ```

2. **Latent Topic Modeling:**
   Identify the topics discussed within the conversation data.

   ```bash
   python3 parse3.py ./conversations.json --topic-modeling --num-topics 5 --top-words 10
   ```

3. **Question Rephrasing Analysis:**
   Identify pairs of similar questions asked in the conversations.

   ```bash
   python3 parse3.py ./conversations.json --rephrasing --similarity-threshold 0.6
   ```

4. **Print Common English Words:**
   Display the top N most common English words.

   ```bash
   python3 parse3.py ./conversations.json --print-english-words 20
   ```

5. **Combining Multiple Analyses:**
   Run multiple analyses together.

   ```bash
   python3 parse3.py ./conversations.json --common-words --filter-english-words --top-n 20 --topic-modeling --num-topics 5 --top-words 10
   ```

**Command-Line Arguments:**

- **`file`:** Path to the conversation JSON file.
- **`--common-words`:** Perform Common Words Analysis.
- **`--topic-modeling`:** Perform Latent Topic Modeling.
- **`--rephrasing`:** Perform Question Rephrasing Analysis.
- **`--print-english-words` [N]:** Print the top N most common English words.
- **`--filter-english-words`:** Filter out top English words from common words analysis.
- **`--top-n` [N]:** Number of top results for common words (default: 20).
- **`--num-topics` [N]:** Number of topics for topic modeling (default: 10).
- **`--top-words` [N]:** Number of words per topic (default: 10).
- **`--similarity-threshold` [FLOAT]:** Threshold for question rephrasing similarity (default: 0.5).

## Example

```bash
python3 parse3.py ./conversations.json --common-words --filter-english-words --top-n 30
```
This command will analyze common words from `conversations.json`, filtering out the top English words, and display the top 30 results.
