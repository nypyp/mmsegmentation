import torch

def init_weight():
    ckpt = torch.load('/home/nypyp/code/mmsegmentation/weights/iter_320000.pth')
    print(type(ckpt))
    if 'state_dict' in ckpt:
        _state_dict = ckpt['state_dict']
        print(type(_state_dict))
        # print(_state_dict)
        for key ,value in _state_dict.items():
            print(key)
init_weight()