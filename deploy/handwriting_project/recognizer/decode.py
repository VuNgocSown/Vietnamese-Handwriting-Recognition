import torch

char_list = [' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'À', 'Á', 'Â', 'Ê', 'Ô', 'Ú', 'Ý', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'Ă', 'ă', 'Đ', 'đ', 'ĩ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư', 'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'Ề', 'ề', 'Ể', 'ể', 'ễ', 'Ệ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'Ở', 'ở', 'ỡ', 'ợ', 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ']

num_classes = len(char_list) + 1
blank_idx = num_classes - 1

def decode_prediction(output_tensor):
    
    output = output_tensor.squeeze(0)  # Đảm bảo shape là (T, C)

    
    pred_indices = torch.argmax(output, dim=1) # Shape: (T)

    # --- Thêm Log chi tiết ---
    print(f"  [decode_prediction] Raw pred_indices (shape {pred_indices.shape}): {pred_indices.cpu().numpy()}")
 

    decoded_text = []
    prev_idx = blank_idx

    for idx in pred_indices:
        idx = idx.item()
        
        is_valid_char_idx = 0 <= idx < len(char_list)

        if idx != blank_idx and idx != prev_idx:
            if is_valid_char_idx:
                decoded_text.append(char_list[idx])
            else:
               
                print(f"  [decode_prediction] Warning: Invalid char index {idx} found.")
        prev_idx = idx

    result = ''.join(decoded_text)
    print(f"  [decode_prediction] Decoded result: '{result}'") # Log kết quả cuối cùng của decode
    return result