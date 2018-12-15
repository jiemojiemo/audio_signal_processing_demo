from scipy.io.wavfile import read
import numpy as np

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def is_power2(num):
    return ((num & (num - 1)) == 0) and num != 0

def wavread(filename):
    fs, x = read(filename)
    
    x = np.float32(x)/norm_fact[x.dtype.name]
    return fs, x

def zero_phase_windowing(x, N):
    '''
    x(numpy array): 输入信号
    N(int): 补零后的大小
    '''
    
    M = len(x)
    
    if(M >= N):                      # 如果x的长度大于N，不需要补零，直接返回
        return x
    
    buffer = np.zeros(N)
    hM1 = (M+1)//2
    hM2 = M//2
    
    buffer[:hM1] = x[-hM1:]
    buffer[-hM2:] = x[:hM2]
    
    return buffer

def undo_zero_phase_windowing(buffer, M):
    
    hM1 = (M+1)//2
    hM2 = M//2
    
    y = np.zeros(M)
    y[:hM2] = buffer[-hM2:]                                        # 零相位窗口逆操作
    y[-hM1:] = buffer[:hM1]
    
    return y