from typing import Any
import torch


class SurrogateHeaviside(torch.autograd.Function):
    # Activation function with surrogate gradient
    sigma = 10.0
    gradient_method = torch.sigmoid
    # gradient_method = torch.tanh

    @staticmethod
    def forward(ctx, input: Any) -> Any:
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = (
            grad_input
            * SurrogateHeaviside.gradient_method(SurrogateHeaviside.sigma * input)
            * SurrogateHeaviside.gradient_method(-SurrogateHeaviside.sigma * input)
        )
        return grad
