import gzip
import numpy as np

def main():
    data = 0

    #gzipファイルからMNISTデータを呼び出して2次元の形にする
    with gzip.open('./train-images-idx3-ubyte.gz', 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape([-1,28,28])
    data = np.where(data < 102, -1, 1000)

    for e,imgtmp in enumerate(data):
        img = np.pad(imgtmp,[(2,2),(2,2)],"constant",constant_values=(-1))
        cnt = 0

        for i in range(2,30):
            for j in range(2,30):
                if img[i][j] == 1000:
                    img[i][j] = cnt
                    cnt+=1
        
        edges = []
        # y座標、x座標
        npzahyou = np.zeros((cnt,2))

        for i in range(2,30):
            for j in range(2,30):
                if img[i][j] == -1:
                    continue

                #8近傍に該当する部分を抜き取る。
                filter = img[i-2:i+3,j-2:j+3].flatten()
                filter1 = filter[[6,7,8,11,13,16,17,18]]

                npzahyou[filter[12]][0] = i-2
                npzahyou[filter[12]][1] = j-2

                for tmp in filter1:
                    if not tmp == -1:
                        edges.append([filter[12],tmp])

        np.save("../dataset/graphs/"+str(e),edges)
        np.save("../dataset/node_features/"+str(e),npzahyou)

if __name__=="__main__":
    main()