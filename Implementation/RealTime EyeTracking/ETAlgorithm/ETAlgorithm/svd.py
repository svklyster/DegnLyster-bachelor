import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

# svd function

def radius (u, v):

    w = np.float64(0)
    u = math.abs(u)
    v = math.abs(v)
    if (u > v):
        w = v / u
        return (u * math.sqrt(1 + w * w));
    else:
        if (v):
            w = u / v
            return (v * math.sqrt(1 + w * w));
        else:
            return 0.0;



def svd (m, n, a, p, d, q):
    
    flag = 0
    i = 0
    its = 0
    j = 0
    jj = 0
    k = 0
    l = 0
    nm = 0
    nm1 = n - 1
    mm1 = m - 1
    c = 0
    f = 0
    h = 0
    s = 0
    x = 0
    y = 0
    z = 0
    anorm = 0
    g = 0
    scale = 0
    r = np.zeros(n)

    for i in range(m):
        for j in range(n):
            p[i][j] = a[i][j]

    for i in range(n):
        l = i + 1
        r[i] = scale * g
        g = s = scale = 0.0
        if (i < m):
            for k in range(i, m):
                scale += math.abs(p[k][i])
            if (scale):
                for k in range(i, m):
                    p[k][i] /= scale
                    s += p[k][i] * p[k][i]
                f = p[i][i]
                g = -SIGN(math.sqrt(s), f)
                h = f * g - s
                p[i][i] = f - g
                if (i != nm1):
                    for j in range(j, n):
                        s = 0.0
                        for k in range(m):
                            s += p[k][i] * p[k][j]
                        f = s / h
                        for k in range(i, m):
                            p[k][j] += f * p[k][i]
                for k in range(i, m):
                    p[k][i] *= scale
        d[i] = scale * g
        g = s = scale = 0.0
        if (i < m and i != nm1):
            for k in range(l, n):
                scale += math.abs(p[i][k])
            if(scale):
                for k in range(l, n):
                    p[i][k] /= scale
                    s += p[i][k] * p[i][k]
                f = p[i][l]
                g = -SIGN(math.sqrt(s), f)
                h = f * g - s
                p[i][l] = f - g
                for k in range(l, n):
                    r[k] = p[i][k] / h
                if (i != nm1):
                    for j in range(l, m):
                        s = 0.0
                        for k in range(l, n):
                            s += p[j][k] * p[i][k]
                        for k in range(l, n):
                            p[j][k] += s * r[k]
                for k in range(l, n):
                    p[i][k] *= scale
        anorm = MAX(anorm, math.abs(d[i]) + math.abs(r[i]))
        for i in range(n - 1, 0, -1):
            if(i < nm1):
                if(g):
                    for j in range(l, n):
                        q[j][i] = (p[i][j] / p[i][l]) / g
                    for j in range(l, n):
                        s = 0.0
                        for k in range(l, n):
                            s += p[i][k] * q[k][j]
                        for k in range(l, n):
                            q[k][j] += s * q[k][i]
                for j in range(l, n):
                    q[i][j] = q[j][i] = 0.0
            q[i][i] = 1.0
            g = r[i]
            l = i
        for i in range(n - 1, 0, -1):
            l = i + 1
            g = d[i]
            if(i < nm1):
                for j in range(l, n):
                    p[i][j] = 0.0
            if(g):
                g = 1 / g
                if(i != nm1):
                    for j in range(l, n):
                        s = 0.0
                        for k in range(l, m):
                            s += p[k][i] * p[k][j]
                        f = (s / p[i][i]) * g
                        for k in range(i, m):
                            p[k][j] += f * p[k][i]
                for j in range(i, m):
                    p[j][i] *= g
            else:
                for j in range(i, m):
                    p[j][i] = 0.0
            ++p[i][i]
        for k in range(n - 1, 0, -1):
            its = 0
            for its in range(30):
                flag = 1
                for l in range(k, 0, -1):
                    nm = l - 1
                    if(math.abs(r[l]) + anorm == anorm):
                        flag = 0
                        break;
                    if(math.abs(d[nm]) + anorm == anorm):
                        break;
                if(flag):
                    c = 0.0
                    s = 1.0
                    for i in range(l, k):
                        f = s * r[i]
                        if(math.abs(f) + anorm != anorm):
                            g = d[i]
                            h = radius(f, g)
                            d[i] = h
                            h = 1.0 / h
                            c = g * h
                            s = -f * h
                            for j in range(m):
                                y = p[j][nm]
                                z = p[j][i]
                                p[j][nm] = y * c + z * s
                                p[j][i] = z * c - y * s
                z = d[k]
                if(l == k):
                    if(z < 0.0):
                        d[k] = -z
                        for j in range(n):
                            q[j][k] = (-q[j][k])
                    break;
                if (its == 30):
                    return;
                x = d[l]
                nm = k - 1
                y = d[nm]
                g = r[nm]
                h = r[k]
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y)
                g = radius(f, 1.0)
                f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x
                c = s = 1.0
                for j in range(l, nm):
                    i = j + 1
                    g = r[i]
                    y = d[i]
                    h = s * g
                    g = c * g
                    z = radius(f, h)
                    r[j] = z
                    c = f / z
                    s = h / z
                    f = x * c + g * s
                    g = g * c - x * s
                    h = y * s
                    y = y * c
                    for jj in range(n):
                        x = q[jj][j]
                        z = q[jj][i]
                        q[jj][j] = x * c + z * s
                        q[jj][i] = z * c - x * s
                    z = radius(f, h)
                    d[j] = z

                    if(z):
                        z = 1.0 / z
                        c = f * z
                        s = h * z

                    f = (c * g) + (s * y)
                    x = (c * y) - (s * g)
                    for jj in range(m):
                        y = p[jj][j]
                        z = p[jj][i]
                        p[jj][j] = y * c + z * s
                        p[jj][i] = z * c - y * s
                r[l] = 0.0
                r[k] = f
                d[k] = x
        #free stuff
                



            





















        
    
