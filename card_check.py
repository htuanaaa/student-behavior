import torch

def get_device():
    return 0 if torch.cuda.is_available() else 'cpu'

def get_device_name():
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"

if __name__ == "__main__":
    print("có card rời không?", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("tên card đang dùng:", torch.cuda.get_device_name(0))
    else:
        print("vẫn đang chạy bằng cpu")

