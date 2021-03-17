#楽しい色変換

def rgb_to_yiq(_df, r, g, b):
    _df["color_YIQ_y"] = 0.30*r + 0.59*g + 0.11*b
    _df["color_YIQ_i"] = 0.74*(r-_df["color_YIQ_y"]) - 0.27*(b-_df["color_YIQ_y"])
    _df["color_YIQ_q"] = 0.48*(r-_df["color_YIQ_y"]) + 0.41*(b-_df["color_YIQ_y"])
    
    return _df

def rgb_to_xyz(_df, r, g, b):
    _df["color_XYZ_x"] = 0.3811*r + 0.5783*g + 0.0402*b
    _df["color_XYZ_y"] = 0.1967*r + 0.7244*g + 0.0782*b
    _df["color_XYZ_z"] = 0.0241*r + 0.1288*g + 0.8444*b
    
    return _df

def rgb_to_gray(_df, r, g, b):
    _df["color_gray"] = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return _df

def rgb_to_hsv(r: int, g: int, b: int):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx) * 100
    v = mx * 100
    return tuple([h, s, v])

def rgb_to_hls(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    # XXX Can optimize (maxc+minc) and (maxc-minc)
    l = (minc+maxc)/2.0
    if minc == maxc:
        return 0.0, l, 0.0
    if l <= 0.5:
        s = (maxc-minc) / (maxc+minc)
    else:
        if 2.0-maxc-minc == 0:
            s = 0
        else:
            s = (maxc-minc) / (2.0-maxc-minc)
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return tuple([h, l, s])