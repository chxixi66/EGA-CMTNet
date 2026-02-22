import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LearnableCanny(nn.Module):
    def __init__(self, channel=1, initial_sigma=2.2, initial_weight=0.0):
        super(LearnableCanny, self).__init__()
        self.channel = channel
       
        self.sigma1 = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32))
        self.enhance_weight1 = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float32))
        self.sigma2 = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32))
        self.enhance_weight2 = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float32))
    
    def forward(self, img1 ,img2):
      
        sigma1 = 1 - torch.sigmoid(self.sigma1)  
        enhance_weight1 = torch.sigmoid(self.enhance_weight1)
        sigma2 = 1 - torch.sigmoid(self.sigma2)
        enhance_weight2 = torch.sigmoid(self.enhance_weight2)

        m1 =  canny_conv(img1, self.channel, sigma1, enhance_weight1)
        
        m2 = canny_conv(img2, self.channel, sigma2, enhance_weight2)

        f = (1 - m1) * (1 - m2) + m1 * m2

        return 1-f,m1,m2
       
def sobel_conv(data, channel=1):
    device = data.device
    conv_op_x = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [ 0, 0, 0],
                                   [ 1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5 * abs(edge_x) + 0.5 * abs(edge_y)
    return result

def canny_conv(data, channel=1, sigma=0.1, enhance_weight=0.5):
    data = data.to(device)
    

    blurred = adaptive_smoothing_filter(data, k=10, iter_num=3)

    sobel_edges = sobel_conv(blurred, channel=channel)
    median_value = median_threshold(sobel_edges)
        
    low_threshold = (1.0 - sigma) * median_value
    high_threshold = (1.0 + sigma) * median_value
        
    low_threshold = max(0.001, min(low_threshold, 1.0))
    high_threshold = max(0.001, min(high_threshold, 1.0))
    
  
    grad_kernels = {
    'x': torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device),  
    'y': torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device),  
    '45': torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32, device=device),  
    '135': torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32, device=device)  
    }

    conv_ops = {}
    for name, kernel in grad_kernels.items():
      kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
      conv = nn.Conv2d(channel, channel, (3, 3), stride=1, padding=1, groups=channel, bias=False).to(device)
      conv.weight.data = kernel
      conv_ops[name] = conv


    edge_x = conv_ops['x'](blurred)    
    edge_y = conv_ops['y'](blurred)   
    edge_45 = conv_ops['45'](blurred)  
    edge_135 = conv_ops['135'](blurred)


    magnitude = torch.sqrt(edge_x**2 + edge_y**2 + edge_45**2 + edge_135**2)
    angle = torch.atan2(edge_y, edge_x) % torch.pi
   
    suppressed = torch.zeros_like(magnitude)
    
    pad_magnitude = nn.functional.pad(magnitude, (1, 1, 1, 1), mode='replicate')
    
    dir_angles = torch.tensor([0, torch.pi/4, torch.pi/2, 3*torch.pi/4], device=device)
    theta_expanded = angle.unsqueeze(-1)
    dir_dist = torch.min(
        torch.abs(theta_expanded - dir_angles), 
        torch.pi - torch.abs(theta_expanded - dir_angles)
    )
    dir_weights = F.softmax(-dir_dist / 0.01, dim=-1)
    
    B, C, H, W = magnitude.shape
   
    dir_neighbors = [
        (pad_magnitude[:, :, 1:-1, :-2], pad_magnitude[:, :, 1:-1, 2:]),    
        (pad_magnitude[:, :, :-2, :-2], pad_magnitude[:, :, 2:, 2:]),     
        (pad_magnitude[:, :, :-2, 1:-1], pad_magnitude[:, :, 2:, 1:-1]),    
        (pad_magnitude[:, :, :-2, 2:], pad_magnitude[:, :, 2:, :-2])       
    ]
    
    for dir_idx in range(4):
        neigh1, neigh2 = dir_neighbors[dir_idx]
      
        neigh1 = F.interpolate(neigh1, size=(H, W), mode='nearest') if neigh1.shape[2:] != (H, W) else neigh1
        neigh2 = F.interpolate(neigh2, size=(H, W), mode='nearest') if neigh2.shape[2:] != (H, W) else neigh2
        
        all_mags = torch.cat([magnitude.unsqueeze(-1), neigh1.unsqueeze(-1), neigh2.unsqueeze(-1)], dim=-1)
        soft_weights = F.softmax(all_mags / 0.1, dim=-1)[..., 0]
        suppressed += magnitude * soft_weights * dir_weights[..., dir_idx]
    
    max_magnitude = torch.max(suppressed)
    if max_magnitude > 0:
        suppressed = suppressed / max_magnitude
    
    Shigh = F.hardswish(suppressed - high_threshold)
    Slow = F.hardswish(suppressed - low_threshold)
    ECanny = suppressed * (Shigh + Slow)
   
    dilation_kernel = torch.ones(1, 1, 3, 3, device=device)
    high_continuous = F.hardswish(suppressed - high_threshold)
    dilated_high = F.conv2d(high_continuous, dilation_kernel, stride=1, padding=1, groups=channel)
    weak_contribution = dilated_high * F.hardswish(suppressed - low_threshold) * (1 - high_continuous)
    result = ECanny + weak_contribution
    
    result = enhance_edges(result, data, weight=enhance_weight)
    result_min = result.min()
    result_max = result.max()
   
    denom = result_max - result_min
    result = (result - result_min) / denom
    
    return result

def adaptive_smoothing_filter(f, k=10, iter_num=3):
    f = f.clone()
    batch = int(f.shape[0])
    channel = int(f.shape[1])
    height = int(f.shape[2])
    width = int(f.shape[3])
    
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),  (0, 0),  (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    for _ in range(iter_num):
        f_right = F.pad(f, (0, 0, 0, 1), mode='replicate')[:, :, 1:, :]
        f_left = F.pad(f, (0, 0, 1, 0), mode='replicate')[:, :, :-1, :]
        Gx = (f_right - f_left) / 2
        
        f_down = F.pad(f, (0, 1, 0, 0), mode='replicate')[:, :, :, 1:]
        f_up = F.pad(f, (1, 0, 0, 0), mode='replicate')[:, :, :, :-1]
        Gy = (f_down - f_up) / 2
        
        grad_sq = Gx ** 2 + Gy ** 2
        weight = torch.exp(-grad_sq / (2 * k ** 2))
        
        f_new = torch.zeros_like(f)
        weight_sum = torch.zeros_like(f)
        
        for dx, dy in neighbors:
            start_h = int(max(0, -dx))
            end_h = int(min(height, height - dx))
            start_w = int(max(0, -dy))
            end_w = int(min(width, width - dy))
            
            f_neigh = f[:, :, start_h:end_h, start_w:end_w]
            w_neigh = weight[:, :, start_h:end_h, start_w:end_w]
            
            pad_h = (max(0, dx), max(0, -dx))
            pad_w = (max(0, dy), max(0, -dy))
            f_neigh = F.pad(f_neigh, pad_w + pad_h, mode='replicate')
            w_neigh = F.pad(w_neigh, pad_w + pad_h, mode='replicate')
            
            f_new += f_neigh * w_neigh
            weight_sum += w_neigh
        
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        f = f_new / weight_sum
    
    return f

def median_threshold(image_tensor):

    median_values = torch.median(image_tensor.view(image_tensor.shape[0], image_tensor.shape[1], -1), dim=2)[0]

    return torch.mean(median_values).item()

def enhance_edges(canny_edges, image_tensor, weight=0.5):
   
    channel = image_tensor.size(1)
    sobel_edges = sobel_conv(image_tensor, channel=channel)

    enhanced_edges = (1 - weight) * canny_edges + weight * sobel_edges
    
    return enhanced_edges

if __name__ == "__main__":
    a = torch.randn(1, 1, 640, 480)
    b = torch.randn(1, 1, 640, 480)
    
    m1 = canny_conv(a)
    m2 = canny_conv(b)
   
    canny_result = 1 - ((1 - m1) * (1 - m2) + m1 * m2)
    print(f"Canny shape: {canny_result.shape}")




