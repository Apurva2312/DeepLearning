import numpy as np


# method to calculate L infinity, L1 and L2 norms
def calculate_norms(a, b):
    L_inf_norm = np.max(np.abs(a - b))
    L1_norm = np.sum(np.abs(a - b))
    L2_norm = np.sqrt(np.sum((a - b)**2))

    return L_inf_norm, L1_norm, L2_norm


if __name__ == '__main__':
    # 100D feature vector
    A = np.random.rand(100, 1)
    B = np.random.rand(100, 1)
    L_inf, L1, L2 = calculate_norms(A, B)

    print("L∞ norm feature distance between vectors A and B is : {}".format(L_inf))
    print("L1 norm feature distance between vectors A and B is : {}".format(L1))
    print("L2 norm feature distance between vectors A and B is : {}".format(L2))

