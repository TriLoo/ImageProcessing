#include "gff.h"

GFF::GFF(int wid, int hei, int lr, int gr) : WMap(wid, hei, lr, gr)
{
}

GFF::~GFF()
{
}

__global__ void multAddTwo_kernel(float *d_out, float *d_inA, float *d_inB, float *d_inC, float *d_inD, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;
    d_out[offset] = __fmul_rd(d_inA[offset], d_inB[offset]) + __fmul_rd(d_inC[offset], d_inD[offset]);
}
__global__ void multAddTri_kernel(float *d_out, float *d_in, float *d_inA, float *d_inB, float *d_inC, float *d_inD, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;
    d_out[offset] = d_in[offset] + __fmul_rd(d_inA[offset], d_inB[offset]) + __fmul_rd(d_inC[offset], d_inD[offset]);
}

void GFF::gffFusion(float *d_imgOut, float *d_imgInA, float *d_imgInB, int wid, int hei, int tsr, int lr, int gr,
                    double gsig, int guir, double guieps)
{
    float *d_tempA, *d_tempB, *d_tempC, *d_tempD;
    float *d_tempE, *d_tempF, *d_tempG, *d_tempH;
    cudaCheckErrors(cudaMalloc((void **)&d_tempA, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempB, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempC, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempD, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempE, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempF, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempG, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempH, sizeof(float) * wid * hei));

    twoscale(d_tempA, d_tempB, d_imgInA, wid, hei, 15);
    twoscale(d_tempC, d_tempD, d_imgInB, wid, hei, 15);
    vector<float *> d_imgInS(0);
    d_imgInS.push_back(d_tempA);
    d_imgInS.push_back(d_tempB);
    d_imgInS.push_back(d_tempC);
    d_imgInS.push_back(d_tempD);

    //weightedmap(d_tempC, d_tempD, d_imgInA, d_imgInB, wid, hei, 1, 5, 0.1, 10, 0.1);
    weightedmap(d_tempE, d_tempF, d_tempG, d_tempH, d_imgInA, d_imgInB, wid, hei, lr, gr, gsig, guir, guieps);
    vector<float *> d_wmapInS(0);
    d_wmapInS.push_back(d_tempE);
    d_wmapInS.push_back(d_tempF);
    d_wmapInS.push_back(d_tempG);
    d_wmapInS.push_back(d_tempH);

    Merge(d_imgOut, d_imgInS, d_wmapInS, wid, hei);

    cout << "In gffFusion low pass part: " << endl;
    cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaFree(d_tempA);
    cudaFree(d_tempB);
    cudaFree(d_tempC);
    cudaFree(d_tempD);
    cudaFree(d_tempE);
    cudaFree(d_tempF);
    cudaFree(d_tempG);
    cudaFree(d_tempH);
}

void GFF::gffFusionTest(float *imgOut, float *imgInA, float *imgInB, int wid, int hei, int tsr, int lr, int gr,
                        double gsig, int guir, double guieps)
{
    float *d_imgInA, *d_imgOut, *d_imgInB;
    cudaCheckErrors(cudaMalloc((void **)&d_imgInA, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgInB, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_imgOut, 0, sizeof(float) * wid * hei));

    cudaCheckErrors(cudaMemcpy(d_imgInA, imgInA, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_imgInB, imgInB, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    gffFusion(d_imgOut, d_imgInA, d_imgInB, wid, hei, 15, 1, 5, 0.1, 10, 0.1);

    //cudaDeviceSynchronize();
    cudaCheckErrors(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * hei * wid, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_imgInA);
    cudaFree(d_imgOut);
    cudaFree(d_imgInB);
}

void GFF::Merge(float *d_imgOut, vector<float *> &d_imgInS, vector<float *> &d_wmapInS, const int wid, const int hei)
{
    float *d_temp;
    cudaCheckErrors(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + BLOCKSIZE - 1) / BLOCKSIZE;
    blockPerGrid.y = (hei + BLOCKSIZE - 1) / BLOCKSIZE;
    multAddTwo_kernel<<<blockPerGrid, threadPerBlock>>>(d_temp, d_imgInS[0], d_wmapInS[0], d_imgInS[1], d_wmapInS[1], wid, hei);  // Base layer
    multAddTri_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOut, d_temp, d_imgInS[2], d_wmapInS[2], d_imgInS[3], d_wmapInS[3], wid, hei);  // Detail layer

    cudaDeviceSynchronize();
    cudaFree(d_temp);
}

void GFF::MergeTest(float *imgOut, vector<float *> &imgInS, vector<float *> &wampInS, int wid, int hei)
{
}

