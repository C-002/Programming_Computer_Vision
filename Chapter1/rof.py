import numpy as np

def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """
        使用A.Chambolle(2005)在公式(11)中的计算步骤实现Rudin-Osher-Fatemi(ROF)去噪模型
        输入：含有噪声的输入图像（灰度图像）、U的初值、TV正则项权值、步长、停止条件
        输出：去噪和去除纹理后的图像、纹理残留
    """

    m, n = im.shape

    U = U_init
    Px = im #对偶域的x分量
    Py = im #对偶域的y分量
    error = 1

    while(error > tolerance):
        Uold = U
        #原始变量的梯度
        GradUx = np.roll(U, -1, axis=1)-U
        GradUy = np.roll(U, -1, axis=0)-U

        #更新对偶变量
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew #更新x分量（对偶）
        Py = PyNew/NormNew #更新y分量（对偶）

        #更新原始变量
        RxPx = np.roll(Px, 1, axis=1) #对x分量进行向右x轴平移
        RyPy = np.roll(Py, 1, axis=0) #对y分量进行向右y轴平移

        DivP = (Px-RxPx) + (Py-RyPy) #对偶域的散度
        U = im + tv_weight*DivP #更新原始变量
        
        # 更新误差
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m)

    return U, im-U
