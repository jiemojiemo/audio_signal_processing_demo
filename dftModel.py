import numpy as np
import math
from scipy.fftpack import fft, ifft
from scipy.signal import get_window
import utilFunctions as UF

tol = 1e-14

def dftAnal(x, w, N):
    '''
    x(numpy array): 输入信号
    w(numpy array): 窗函数生成的信号
    N(int): FFT大小
    '''
    
    if not(UF.is_power2(N)):
        raise ValueError('FFT size(N) is not a power of 2')           # FFT大小必须为2的次幂
        
    if(w.size > N):
        raise ValueError('Window size(M) is bigger than FFT size')      # 窗大小必须小于等于FFT大小
        
        
    # 加窗 
    w = w/sum(w)                                         # 对窗函数进行归一化
    xw = x*w                                           
    
    # zero-phase windowing
    fftbuffer = UF.zero_phase_windowing(xw, N)                     # 零相位窗 + 补零
    
    X = fft(fftbuffer)                                     # FFT
    hN = (N//2)+1                                        # 一半大小（包含0），因为FFT是对称的，因此我们只需要一半数据即可
    absX = np.abs(X[:hN])
    absX[ absX<np.finfo(float).eps ] = np.finfo(float).eps             # 加上eps，防止在log时出错
    mX = 20*np.log10(absX)                                  # 计算分贝幅度
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0
    pX = np.unwrap(np.angle(X[:hN]))
    
    return mX, pX

def dftSynth(mX, pX, M):
    '''
    用DFT合成信号
    mX(numpy array):分贝幅度
    pX(numpy array):相位
    M(int):窗函数大小
    '''
    
    hN = mX.size                                           # 频谱的一半大小（包括位置0）
    N = (hN-1)*2                                           # FFT 大小
    if not(is_power2(N)):
        raise ValueError('size of mX is not (N/2)+1')                 # FFT大小必须为2的次幂
        
    Y = np.zeros(N, dtype=complex)
    
    Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                           # 从mX和pX中得到一个复数即DFT的结果，正半边
    Y[hN:] = 10**(mX[-2:0:-1]/20) * np.exp(1j*pX[-2:0:-1])               # 从mX和pX中得到一个复数即DFT的结果，负半边
                                                       # （由对称性可知，负半边不包括倒数第一个点）
    fftbuffer = np.real(ifft(Y))                                # IFFT
    
    y = UF.undo_zero_phase_windowing(fftbuffer, M)                    # 零相位窗口逆操作
    
    return y

def dftModel(x, w, N):
    mX, pX = dftAnal(x, w, N)
    y = dftSynth(mX, pX, w.size)
    return y