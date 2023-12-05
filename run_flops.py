import torch
import os
from flops import count_model_param_flops

file_prefix = 'logs_flops_'
model_ls = ['18', 'cnn']
method_ls = ['dense', 'set', 'snip', 'grasp', 'rigl']

for model in model_ls:
    for method in method_ls:
        file_name = file_prefix + model + '_' + method
        files = os.listdir(file_name)
        model_file = []
        for file in files:
            if os.path.splitext(file)[1] == '.pth':
                model_file.append(file)
        best_checkpoint = sorted(model_file)[-1]
        model_path = file_name+'/' + best_checkpoint
        print('model_path', model_path)
        model_p = torch.load(model_path)
        x = count_model_param_flops(model=model_p)