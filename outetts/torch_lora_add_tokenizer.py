from transformers import AutoTokenizer
import json

# Load the base tokenizer
tokenizer = AutoTokenizer.from_pretrained("OuteAI/OuteTTS-0.3-500M", trust_remote_code=True)

import json
with open("OuteTTS-0.3-500M-zh/added_tokens.json", "r", encoding="utf-8") as f:
    added_token_dict = json.load(f)

# Get the list of newly added tokens (the keys of the dict are token strings)
added_tokens = list(added_token_dict.keys())

# Add these tokens
num_added = tokenizer.add_tokens(added_tokens)
print(f"✅ Added {num_added} tokens")

# added = tokenizer.add_tokens(["<my_special_token_123456>"])
# print("新增 token 数:", added)
tokenizer.save_pretrained("tokenizer_with_lora_tokens")

#tokenizer.save_pretrained("tokenizer_with_lora_tokens")
#
"""
You should resize the model's token embeddings to match the new tokenizer size.
model.resize_token_embeddings(len(tokenizer))

Another way
from transformers import AddedToken
tokenizer.add_tokens([AddedToken("<speaker1>", lstrip=True, rstrip=False)])

# use next time
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer_with_lora_tokens", trust_remote_code=True)
"""