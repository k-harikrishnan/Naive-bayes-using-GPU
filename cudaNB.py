import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpu_array
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time

class CudaNB:
    def matMultiply(self, a, b):
        if(len(a.shape) != 2 or len(b.shape) != 2 ) :
            print("Input data are not matrices")
            return

        if(a.shape[1] != b.shape[0]):
            print("Dimensions missmatch")
            return

        #if(isGPU == True) :
        return self.matMultiplyGPU(a, b)
        #else :
        #    return self.matMultiplyCPU(a, b)


    def matMultiplyCPU(self, a, b):
        return np.matmul(a, b)

    def matMultiplyGPU(self, a, b):

        rows = a.shape[0]
        cols = b.shape[1]

        commonDim = a.shape[1]


        multiplyfun = SourceModule("""
                        #include <cstdio>
                        __global__ void matMultiply(int * a, int * b, int * c, int commonVal, int X, int Y)
                        {
                            int xPos = threadIdx.x + blockIdx.x * blockDim.x;
                            int yPos = threadIdx.y + blockIdx.y * blockDim.y;

                            if(xPos < X && yPos < Y)
                            {
                                int sum = 0;
                                for(int i = 0; i < commonVal; i++)
                                {
                                    //printf("%d, %d, %d, %d, %d\\n", i, xPos, yPos, a[xPos * commonVal + i], b[i * commonVal + yPos]);
                                    sum += a[xPos * commonVal + i] * b[i * Y + yPos];
                                }

                                c[xPos * Y + yPos] = sum; 
                            }
                        }
                        """)

        matMulti = multiplyfun.get_function("matMultiply")

        c = np.zeros((rows, cols)).astype(np.int32)

        blockXSize = 32
        blockYSize = 32
        blockSize = (blockXSize, blockYSize, 1)

        if (rows % blockXSize == 0) :
            gridXSize = int(rows/blockXSize)
        else :
            gridXSize = int(rows/blockXSize + 1)

        if (cols % blockYSize == 0) :
            gridYSize = int(cols/blockYSize)
        else :
            gridYSize = int(cols/blockYSize)  + 1

        gridSize = (gridXSize, gridYSize, 1)

        matMulti(cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(commonDim), np.int32(rows), np.int32(cols), block=blockSize, grid=gridSize)

        return c

def matMutliTest(cudanb) :
    a = np.random.randint(50, 100, size=(7210,100)).astype(np.int32)
    b = np.random.randint(25, 300, size=(100,7190)).astype(np.int32)

    #print(a)
    #print(b)
    #np.savetxt("a.csv", a, delimiter=',')
    #np.savetxt("b.csv", b, delimiter=',')

    start = time.time()
    c = cudanb.matMultiply(a,b)
    end = time.time()

    print("GPU Time : {}".format((end - start)* 1000))

    start = time.time()
    c_cpu = cudanb.matMultiplyCPU(a,b)
    end = time.time()

    print("CPU Time : {}".format((end - start) * 1000))

    print("Error : {}".format(sum(sum(c - c_cpu))))


if __name__ == "__main__" :
    cudanb = CudaNB()

    np.set_printoptions(threshold=np.nan)

    matMutliTest(cudanb)
    #print(histoTest(cudanb))

    cuda.stop_profiler()
