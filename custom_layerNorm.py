
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import mylayerNorm_cuda

class myLayerNormFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = mylayerNorm_cuda.forward(input)

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = mylayerNorm_cuda.backward(grad_output, input)

        return grad_input[0]

    
class myLayerNorm(nn.Module):
    def __init__(self):
        super(myLayerNorm, self).__init__()
        
    def forward(self, input):
        return myLayerNormFunction.apply(input)

def verify(device):
    test_cnt = 100
    my_layerNorm = myLayerNorm().to(device)
    torch_layerNorm = nn.LayerNorm(normalized_shape=(128), elementwise_affine=False, eps=1e-5).to(device)

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

    print(f'My LayerNorm forward avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch LayerNorm forward avg time: {sum(torch_time[10:])/test_cnt}s')
    
def verify_backward(device):
    test_cnt = 100
    my_layerNorm = myLayerNorm().to(device)
    torch_layerNorm = nn.LayerNorm(normalized_shape=(128), elementwise_affine=False, eps=1e-5).to(device)
    loss_fn = torch.nn.MSELoss()

    for _ in range(test_cnt):
        in_t_my = torch.rand(size=(64, 128), requires_grad=True).to(device)
        in_t_torch = in_t_my.clone().detach().requires_grad_(True).to(device)
        np.testing.assert_allclose(in_t_my.cpu().detach().numpy(), in_t_torch.cpu().detach().numpy(), rtol=1e-3, atol=1e-5)

        out_torch = torch_layerNorm(in_t_torch)
        target_torch = torch.ones_like(out_torch)
        loss_torch = loss_fn(out_torch, target_torch)
        grad_in_t_torch = torch.autograd.grad(loss_torch, in_t_torch, retain_graph=True)[0]

        out_my = my_layerNorm(in_t_my)
        target_my = torch.ones_like(out_my)
        loss_my = loss_fn(out_my, target_my)
        grad_in_t_my = torch.autograd.grad(loss_my, in_t_my, retain_graph=True)[0]

        np.testing.assert_allclose(grad_in_t_torch.cpu().detach().numpy(), grad_in_t_my.cpu().detach().numpy(), rtol=1e-3, atol=1e-5)

    my_time = []
    torch_time = [] 
    for _ in range(test_cnt+10):
        in_t_my = torch.rand(size=(64, 128), requires_grad=True).to(device)
        in_t_torch = in_t_my.clone().detach().requires_grad_(True).to(device)

        out_torch = torch_layerNorm(in_t_torch)
        target_torch = torch.ones_like(out_torch)
        loss_torch = loss_fn(out_torch, target_torch)
        start_time = time.time()
        grad_in_t_torch = torch.autograd.grad(loss_torch, in_t_torch, retain_graph=True)[0]
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)

        out_my = my_layerNorm(in_t_my)
        target_my = torch.ones_like(out_my)
        loss_my = loss_fn(out_my, target_my)
        start_time = time.time()
        grad_in_t_my = torch.autograd.grad(loss_my, in_t_my, retain_graph=True)[0]
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)

    print(f'My LayerNorm backward avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch LayerNorm backward avg time: {sum(torch_time[10:])/test_cnt}s')



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='LayerNorm Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    verify(device)
    verify_backward(device)

if __name__ == '__main__':
    main()
