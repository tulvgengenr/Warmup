# This file has been changed for education and teaching purpose

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import mylinear_cuda

class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        #output = mylinear_cpp.forward(input, weight)
        output = mylinear_cuda.forward(input, weight)

        return output[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        #grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        grad_input, grad_weight = mylinear_cuda.backward(grad_output, input, weight)

        return grad_input, grad_weight

class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)

def verify(device):    
    test_cnt = 100
    my_linear = myLinear(128, 10).to(device)
    torch_linear = nn.Linear(128, 10).to(device)
    torch_linear.weight.data = my_linear.weight.data
    np.testing.assert_array_equal(torch_linear.weight.data.cpu().numpy(), my_linear.weight.data.cpu().numpy())

    # correctness
    # It is a counter-example that myLinear does not guarantee correctness.
    # But you should guarantee yours:)
    
    for _ in range(test_cnt):
        in_t = torch.rand(size=(64, 128)).to(device)
        out_my = my_linear(in_t).detach()
        out_torch = torch_linear(in_t).detach()
        np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy())
    
    # time
    my_time = []
    torch_time = []
    
    for _ in range(test_cnt+10):
        in_t = torch.rand(size=(64, 128)).to(device)
        # my linear
        start_time = time.time()
        out_my = my_linear(in_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)
        
        # torch linear
        start_time = time.time()
        out_torch = torch_linear(in_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)
    print(f'My Linear avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch Linear avg time: {sum(torch_time[10:])/test_cnt}s')

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
