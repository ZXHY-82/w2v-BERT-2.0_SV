import json

with open('./config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# add hard_concrete params
data['limit_l'] = -0.1
data['limit_r'] = 1.1
data['temperature'] = 2/3

# ffn module
layer_num = data['num_hidden_layers']
data['intermediate_size_group'] = [[data['intermediate_size'],data['intermediate_size']] for _ in range(layer_num)]
data['use_feed_forward'] = [[True, True] for _ in range(layer_num)]
# conv module
data['conv_group'] = [[data['hidden_size'],data['hidden_size']] for _ in range(layer_num)]

# attention module
data['num_attention_heads_group'] = [data['num_attention_heads'] for _ in range(layer_num)]
data['use_attention'] = [True for _ in range(layer_num)]

data['prune'] = True

with open('config_prune_stu.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

data['prune'] = False

with open('config_prune_tea.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)