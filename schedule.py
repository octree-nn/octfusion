import numpy as np
import math

def cosine_schedule(t, start = 0, end=1, tau=1, clip_min = 1e-9):
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)

t = 0.5
print(cosine_schedule(t))