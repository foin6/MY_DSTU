def window_partition(x, window_size): # [batch_size, H, W, dim]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows # [batch_size*total_windows_num, win_size_H, win_size_W, dim]