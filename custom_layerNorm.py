
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import mylayerNorm_cuda

class myLayerNormFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, eps):
        ctx.save_for_backward(input, eps)
        #output = input.mm(weight.t())
        #output = mylinear_cpp.forward(input, weight)
        output = mylayerNorm_cuda.forward(input, eps)

        return output[0]
    
class myLayerNorm(nn.Module):
    def __init__(self, eps):
        super(myLayerNorm, self).__init__()
        self.eps = eps
        
    def forward(self, input):
        return myLayerNormFunction.apply(input, self.eps)

def verify(device):
    test_cnt = 100
    my_layerNorm = myLayerNorm(eps=1e-05).to(device)
    torch_layerNorm = nn.LayerNorm(normalized_shape=(128), elementwise_affine=False, eps=1e-5).to(device)
    np.testing.assert_array_equal(torch_layerNorm.eps, my_layerNorm.eps)

    for _ in range(test_cnt):
        in_t = torch.rand(size=(64, 128)).to(device)
        out_my = my_layerNorm(in_t).detach()
        out_torch = torch_layerNorm(in_t).detach()
        np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), rtol=1e-3, atol=1e-5)

    my_time = []
    torch_time = []

    for _ in range(test_cnt+10):
        in_t = torch.rand(size=(64, 128)).to(device)
        # my layerNorm
        start_time = time.time()
        out_my = my_layerNorm(in_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)

        # torch layerNorm
        start_time = time.time()
        out_torch = torch_layerNorm(in_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)

    print(f'My LayerNorm avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch LayerNorm avg time: {sum(torch_time[10:])/test_cnt}s')
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    verify(device)

if __name__ == '__main__':
    main()