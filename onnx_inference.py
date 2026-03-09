import onnxruntime as ort
import cv2

def prepare_input(rgb_image, input_size):
    h, w = rgb_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    
    # Padding to input_size
    padding = [123.675, 116.28, 103.53]  # Same as mean values used in normalization
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    
    # Add padding
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, 
                             pad_w_half, pad_w - pad_w_half, 
                             cv2.BORDER_CONSTANT, value=padding)
    
    # Record padding info for later removal
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    
    # Prepare ONNX input format (NCHW)
    onnx_input = {
        "image": np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)
    }
    
    return onnx_input, pad_info




# Create ONNX Runtime session with CUDA provider
providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "0", "device_id": "0"})]
ort_session = ort.InferenceSession(onnx_model, providers=providers)
 
# Run inference
outputs = ort_session.run(None, onnx_input)
depth = outputs[0].squeeze()  # [H, W]
 
# Remove padding and resize to original shape
depth = depth[pad_info[0]:input_size[0]-pad_info[1], pad_info[2]:input_size[1]-pad_info[3]]
depth = cv2.resize(depth, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)