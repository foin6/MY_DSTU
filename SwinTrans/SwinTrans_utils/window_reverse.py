def window_reverse(windows, window_size, H, W): # windows.shape = [batch_size*total_windows_num, win_size_H, win_size_W, dim]
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size)) # 计算出batch_size
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1) # [batch_size, win_num_H, win_numW, win_size_H, win_size_W, dim]
    # permute ---> [batch_size, win_num_H, win_size_H, win_numW, win_size_W, dim]
    # view ---> [batch_size, Hp, Wp, dim]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x # [batch_size, Hp, Wp, dim]