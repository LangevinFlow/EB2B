import numpy as np
from .util import kernel_move
import cv2
import torch
import sys
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# noise
def make_gradient_filter():
    filters = np.zeros([4, 3, 3], dtype=np.float32)
    filters[0,] = np.array([[0, -1, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

    filters[1,] = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

    filters[2,] = np.array([[0, 0, 0],
                            [-1, 1, 0],
                            [0, 0, 0]])

    filters[3,] = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [-1, 0, 0]])

    return torch.from_numpy(filters).cuda()



# kernel
def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level, move_x, move_y):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise, move_x, move_y)

    return kernel


def gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise, move_x, move_y):
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    raw_kernel_moved = kernel_move(raw_kernel, move_x, move_y)

    # Normalize the kernel and return
    kernel = raw_kernel_moved / np.sum(raw_kernel_moved)
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    return kernel



def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated



def gen_kernel_motion_fixed(k_size, sf, lens, theta, noise):

    # kernel_size = min(sf * 4 + 3, 21)
    kernel_size = k_size[0]
    M = int((sf * 3 + 3) / 2)
    kernel_init = np.zeros([min(sf * 4 + 3, 21), min(sf * 4 + 3, 21)])
    # kernel_init[M-1:M+1,M-len:M-len] = 1
    kernel_init[M:M + 1, M - lens:M + lens + 1] = 1
    kernel = kernel_init + noise
    center = ((sf * 3 + 3) / 2, (sf * 3 + 3) / 2)
    kernel = rotate(kernel, theta, center, scale=1.0)

    kernel = kernel / np.sum(kernel)

    return kernel



def gen_kernel_random_motion(k_size, scale_factor, lens, noise_level):
    # lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    # lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * 360  # np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_motion_fixed(k_size, scale_factor, lens, theta, noise)

    return kernel


def ekp_kernel_generator(U, kernel_size, sf=4, shift='left'):
    '''
    Generate Gaussian kernel according to cholesky decomposion.
    \Sigma = M * M^T, M is a lower triangular matrix.
    Input:
        U: 2 x 2 torch tensor
        sf: scale factor
    Output:
        kernel: 2 x 2 torch tensor
    '''
    #  Mask
    mask = torch.tensor([[1.0, 0.0],
                         [1.0, 1.0]], dtype=torch.float32).to(U.device)
    M = U * mask

    # Set COV matrix using Lambdas and Theta
    INV_SIGMA = torch.mm(M.t(), M)

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = kernel_size // 2 - 0.5 * (sf - 1)
    elif shift.lower() == 'center':
        MU = kernel_size // 2
    elif shift.lower() == 'right':
        MU = kernel_size // 2 + 0.5 * (sf - 1)
    else:
        sys.exit('Please input corrected shift parameter: left , right or center!')

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    Z = torch.stack((X, Y), dim=2).unsqueeze(3).type(torch.float32).to(U.device)  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.permute(0, 1, 3, 2)  # k x k x 1 x 2
    raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))

    # Normalize the kernel and return
    kernel = raw_kernel / torch.sum(raw_kernel)  # k x k
    return kernel.unsqueeze(0).unsqueeze(0)

def extract_kernel_from_resnet(resnet_model, input_tensor, target_size):
    """
    Extract kernel from ResNet features
    Args:
        resnet_model: Pre-trained ResNet model
        input_tensor: Input tensor to ResNet
        target_size: Target kernel size (tuple)
    Returns:
        kernel: Extracted kernel of size target_size
    """
    # Get features from first layer
    features = resnet_model.conv1(input_tensor)
    
    # Average pooling to get kernel-like representation
    kernel = torch.mean(features, dim=1, keepdim=True)
    
    # Resize to target size
    kernel = F.interpolate(kernel, size=target_size, mode='bilinear', align_corners=False)
    
    # Normalize kernel
    kernel = kernel - kernel.min()
    kernel = kernel / kernel.sum()
    
    return kernel.squeeze().cpu().numpy()

def get_resnet_kernel(input_tensor, target_size, device='cuda'):
    """
    Get kernel from pre-trained ResNet
    Args:
        input_tensor: Input tensor
        target_size: Target kernel size (tuple)
        device: Device to run on
    Returns:
        kernel: Extracted kernel
    """
    # Load pre-trained ResNet
    resnet = models.resnet18(pretrained=True)
    resnet = resnet.to(device)
    resnet.eval()
    
    # Extract kernel
    with torch.no_grad():
        kernel = extract_kernel_from_resnet(resnet, input_tensor, target_size)
    
    return kernel


class EmpiricalBayesKernel:
    def __init__(self, k_size, sf, device='cuda'):
        self.k_size = k_size
        self.sf = sf
        self.device = device
        
        # Initialize parameters with requires_grad=True
        self.raw_lambda = nn.Parameter(torch.randn(2, device=device, requires_grad=True))  # raw eigenvalues
        self.raw_theta = nn.Parameter(torch.randn(1, device=device, requires_grad=True))   # rotation angle
        self.raw_dx = nn.Parameter(torch.randn(1, device=device, requires_grad=True) * 0)      # raw x-shift
        self.raw_dy = nn.Parameter(torch.randn(1, device=device, requires_grad=True) * 0)      # raw y-shift
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            self.raw_lambda, self.raw_theta, self.raw_dx, self.raw_dy
        ], lr=5e-2)
        
        # Prior parameters
        self.beta = 1e-4  # prior weight
        
        # Temperature annealing parameters
        self.temp_start = 0.5
        self.temp_end = 1.0
        
    def _get_eigenvalues(self):
        """Get positive eigenvalues with lower bound"""
        return 0.1 + 4.0 * F.softplus(self.raw_lambda)  # λ ∈ [0.1, ∞)
        
    def _get_shifts(self):
        """Get bounded shifts using tanh"""
        shift_range = (self.sf - 1) / 2  # Only allow ±(sf-1)/2
        return shift_range * torch.tanh(torch.stack([self.raw_dx, self.raw_dy]))
        
    def generate_kernel(self):
        """Generate kernel using the complete parameterization"""
        # Get parameters
        lambda_ = self._get_eigenvalues()
        theta = self.raw_theta[0]
        dx, dy = self._get_shifts()
        
        # Compute rotation matrix
        cos, sin = torch.cos(theta), torch.sin(theta)

        R = torch.stack(
                (torch.stack((cos, -sin)),
                torch.stack((sin,  cos)))
            ) 
        # Compute inverse covariance matrix
        #Lambda_inv = torch.diag(1.0 / (lambda_**2))
        Lambda_inv = torch.diag(1.0 / lambda_)
        inv_Sigma = R @ Lambda_inv @ R.t()
        
        # Set mean position (geometric centre + shift, keep graph)
        c = torch.tensor((self.k_size - 1) / 2, device=self.device)
        mu = torch.stack([c + dx, c + dy]).view(2, 1)   # (2,1) tensor
        
        # Create coordinate grid
        X, Y = torch.meshgrid(torch.arange(self.k_size), torch.arange(self.k_size))
        coords = torch.stack([X, Y], dim=-1).float().to(self.device)
        
        # Calculate Gaussian
        zz = coords - mu.t()  # (k,k,2)
        zzT_invS = zz.unsqueeze(-2) @ inv_Sigma  # (k,k,1,2)
        quad = (zzT_invS @ zz.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (k,k)
        raw_kernel = torch.exp(-0.5 * quad)
        
        # Normalize
        kernel = raw_kernel / torch.sum(raw_kernel)
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
    
    def optimize(self, x, y, num_steps=25, current_iter=0):
        """Optimize kernel parameters using Empirical Bayes"""
        # Freeze lambda and theta in early stages
        if current_iter < 0:
            self.raw_lambda.requires_grad = False
            self.raw_theta.requires_grad = False
        else:
            self.raw_lambda.requires_grad = True
            self.raw_theta.requires_grad = True
            
        # Temperature annealing
        temp = self.temp_start + (self.temp_end - self.temp_start) * min(1.0, current_iter / 100)
        
        for _ in range(num_steps):
            self.optimizer.zero_grad()
            
            # Generate kernel
            kernel = self.generate_kernel()
            lambda_ = self._get_eigenvalues()
            
            # Compute data likelihood with reflect padding
            x_pad = F.pad(x, mode='reflect', 
                         pad=(self.k_size//2, self.k_size//2, 
                              self.k_size//2, self.k_size//2))
            out = F.conv2d(x_pad, kernel.expand(3, -1, -1, -1), groups=3)
            out = out[:, :, 0::self.sf, 0::self.sf]
            
            # Compute MSE loss with temperature
            mse_loss = F.mse_loss(out, y) * temp
            
            # Compute inverse eigenvalue penalty
            inv_penalty = self.beta * (1.0 / (lambda_[0]**2) + 1.0 / (lambda_[1]**2))
            
            # Total loss
            total_loss = mse_loss + inv_penalty
            
            # Backpropagate
            total_loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([ self.raw_theta,  self.raw_dx, self.raw_dy], 0.5)
            
            self.optimizer.step()
            
            # Detach to free memory
            total_loss.detach()
            
        # Return the final kernel without gradient information
        with torch.no_grad():
            return self.generate_kernel()
