# WHAT DOES IT DO?

Basically turns plain text into two arrays that the model needs:
- `input_ids` -> integers that represent subword tokens
- `attention_mask` -> 1s theres are real tokens, 0s when there are padding 
*basically where to draw the attention*

## Input 
if you pass `--text "some string"` it encodes that single string and exits
If you pipe lines via stdin `(e.g., echo "hello" | python tools/tokenize.py ...)`, it encodes each line and prints one JSON per line.

We have `chmod` to change the file permissions and +x to add the execute bit to run the script as a program