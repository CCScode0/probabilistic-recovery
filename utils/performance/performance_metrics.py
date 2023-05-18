import numpy as np

def distance_to_strip_center(l, x, a, b):
    dist = [l @ x - (a + b) / 2 ] / np.linalg.norm(l) 
    return abs(dist).tolist()

def in_strip(l, x, a, b):
    if l @ x >= a and l @ x <= b:
        success = 1
    else:
        success = 0
    return success
    