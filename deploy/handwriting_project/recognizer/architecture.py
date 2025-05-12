import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class LSTM_From_Scratch(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, states):
        h_prev, c_prev = states
        combined = torch.cat((x_t, h_prev), dim=1)
        i_t = torch.sigmoid(self.W_i(combined))
        f_t = torch.sigmoid(self.W_f(combined))
        o_t = torch.sigmoid(self.W_o(combined))
        g_t = torch.tanh(self.W_c(combined))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        h_t = self.dropout(h_t)
        return h_t, (h_t, c_t)

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout=0.2):
        super().__init__()
        self.forward_lstm = LSTM_From_Scratch(input_dim, hidden_size, dropout=dropout)
        self.backward_lstm = LSTM_From_Scratch(input_dim, hidden_size, dropout=dropout)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        forward_outputs = []
        for t in range(seq_len):
            h_f, (h_f, c_f) = self.forward_lstm(x[:, t, :], (h_f, c_f))
            forward_outputs.append(h_f)
        forward_outputs = torch.stack(forward_outputs, dim=1)
        h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
        backward_outputs = []
        for t in reversed(range(seq_len)):
            h_b, (h_b, c_b) = self.backward_lstm(x[:, t, :], (h_b, c_b))
            backward_outputs.insert(0, h_b)
        backward_outputs = torch.stack(backward_outputs, dim=1)
        outputs = torch.cat((forward_outputs, backward_outputs), dim=2)
        return outputs

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(3, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.rnn1 = BidirectionalLSTM(1024, 512, dropout=0.2)
        self.rnn2 = BidirectionalLSTM(1024, 512, dropout=0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)

def preprocess(image, resize_max_width=2167, height=118):
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image}")
    else:
        img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    new_w = int(height / h * w)
    img = cv2.resize(img, (new_w, height))

    if new_w < resize_max_width:
        img = np.pad(img, ((0,0), (0, resize_max_width - new_w)), mode='median')
    else:
        img = img[:, :resize_max_width]
    

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 4)
    
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, axis=0)  # Thêm kênh (1, H, W)
    img = np.expand_dims(img, axis=0)  # Thêm batch (1, 1, H, W)
    return img

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )
    return thresh

def remove_noise(thresh, min_area=200):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    output = np.zeros_like(thresh)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255
    return output

def segment_boxes(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    
    original_shape = img.shape
    
    
    thresh = thresholding(img)
    cleaned = remove_noise(thresh)
    
    kernel = np.ones((17, 200), np.uint8)
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    
    
    padding = 12
    box_images, boxes_info = [], []
    for ctr in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img.shape[1])
        y2 = min(y + h + padding, img.shape[0])
        
        boxes_info.append({'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1})
        box_images.append(img[y1:y2, x1:x2])
    
    return box_images, boxes_info, original_shape


def segment_lines(image_path):
    
    return segment_boxes(image_path)