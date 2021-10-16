import  numpy as np
from simhash import hamming_distance,comp_doc
if __name__=='__main__':
    mat_list = np.load("E:/pythonwork/reuslt_hash.npy")
    m = []
    for i in mat_list:
        m.append(str(i))
    a = m[1]
    sim = {}
    for i in range(len(m)):
        sim[i] = (comp_doc(1, i, m))
    t_order = sorted(sim.items(), key=lambda x: x[1], reverse=False)
    print(t_order)

