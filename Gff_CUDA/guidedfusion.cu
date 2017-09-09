#include "guidedfusion.h"

// the input image is stored on host memory
void GFusion::guidedfusion(float *imgOut, const float * __restrict__ imgInA, const float * __restrict__ imgInB, int wid, int hei)
{
    cout << "---------Input Image Information---------" << endl;
    cout << " |-    Width  : " << wid << endl;
    cout << " |-    Height : " << hei << endl;
    cout << "-----------------------------------------" << endl;
    cudaError_t cudaState = cudaSuccess;

    float *d_imgInA, *d_imgInB, *d_imgOut;

    // TODO: allocate below variables on device
    //      d_scaleA_H, d_scaleA_L, d_scaleB_H, d_scaleB_L
    //      d_lapA, d_gauA, d_lapB, d_gauB
    //      d_wmapA, d_wmapB, d_gmapA_H, d_gmapB_H, d_gmapA_L, d_gmapB_L
    //      d_coeff_L, d_coeff_H
    // First Version : allocate all above variables
    float *d_scaleA_H, *d_scaleA_L, *d_scaleB_H, *d_scaleB_L;
    float *d_lapA, *d_gauA, *d_lapB, *d_gauB;
    float *d_wmapA, *d_wmapB, *d_gmapA_H, *d_gmapB_H, *d_gmapA_L, *d_gmapB_L;
    float *d_coeff_L, *d_coeff_H;

    // 1-D dimension version
    // malloc input image on device
    cudaState = cudaMalloc((void **)&d_imgInA, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    // copy visual image data from host to device
    cudaState = cudaMemcpy(d_imgInA, imgInA, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    // infrared image
    cudaState = cudaMalloc((void **)&d_imgInB, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMemcpy(d_imgInB, imgInB, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    // prepare the output image memory
    cudaState = cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);

    // Step 1 : Do two scale decomposing on two input images
    // i.e. get the coefficients of input images in transform domain
    // use the 'global memory version'
    cudaState = cudaMalloc((void **)&d_scaleA_H, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_scaleA_L, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_scaleB_H, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_scaleB_L, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    twoscale(d_scaleA_H, d_scaleA_L, d_imgInA, wid, hei);
    twoscale(d_scaleB_H, d_scaleB_L, d_imgInB, wid, hei);

    cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cout << "Step 1 Success " << endl;


    // Step 2 : Get the weight map
    // i.e. implement the fusion rules
    // create laplacian & gauss filter
    createfilterLG();

    cout << "Create Filter LG Success" << endl;

    // process image A
    cudaState = cudaMalloc((void **)&d_lapA, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_gauA, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    // do laplacian filter
    laplafilter(d_lapA, d_imgInA, wid, hei);
    // do gauss filter
    gaussfilter(d_gauA, d_lapA, wid, hei);

    // process image B
    cudaState = cudaMalloc((void **)&d_lapB, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_gauB, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    // do laplacian filter
    laplafilter(d_lapB, d_imgInB, wid, hei);
    // do gauss filter
    gaussfilter(d_gauB, d_lapB, wid, hei);

    // get the weight map
    cudaState = cudaMalloc((void **)&d_wmapA, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_wmapB, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    weightmap(d_wmapA, d_wmapB, d_gauA, d_gauB, wid, hei);

    //cout << "Weight map generated ..." << endl;

    // TODO: add two more parameters to guidedfilter to adjust its radius & regulation values
    // do guided filter
    cudaState = cudaMalloc((void **)&d_gmapA_H, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_gmapA_L, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);

    //cout << "Weight map generated ..." << endl;
    cout << "Step 1.5 success" << endl;

    // process on image A
    guidedfilter(d_gmapA_L, d_imgInA, d_wmapA, wid, hei);   // low-pass weighted map
    guidedfilter(d_gmapA_H, d_imgInA, d_wmapA, wid, hei);   // high-pass weighted map

    cudaState = cudaMalloc((void **)&d_gmapB_H, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_gmapB_L, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);

    cout << "Step 1.6 success" << endl;

    // process on image B
    guidedfilter(d_gmapB_L, d_imgInB, d_wmapB, wid, hei);
    guidedfilter(d_gmapB_H, d_imgInB, d_wmapB, wid, hei);

    cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cout << "Step 2 success" << endl;

    // Step 3 : Get the final fusion result
    cudaState = cudaMalloc((void **)&d_coeff_H, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_coeff_L, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    // i.e. the inverse process of step 1
    // get the high-pass & low-pass coefficients
    getcoeff(d_coeff_L, d_scaleA_L, d_scaleB_L, d_gmapA_L, d_gmapB_L, wid, hei);
    getcoeff(d_coeff_H, d_scaleA_H, d_scaleB_L, d_gmapA_H, d_gmapB_H, wid, hei);

    // get the final fusion result
    twoscalerec(d_imgOut, d_coeff_L, d_coeff_H, wid, hei);

    // copy fusion results from device back to host
    cudaState = cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    // copy fusion result back
    cudaState = cudaMemcpy(d_imgOut, imgOut, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    cout << "Fusion has been done !" << endl;

    // free memory on device
    cudaFree(d_imgInA);
    cudaFree(d_imgInB);
    cudaFree(d_imgOut);
    cudaFree(d_scaleA_H);
    cudaFree(d_scaleA_L);
    cudaFree(d_scaleB_H);
    cudaFree(d_scaleB_L);
    cudaFree(d_lapA);
    cudaFree(d_lapB);
    cudaFree(d_gauA);
    cudaFree(d_gauB);
    cudaFree(d_wmapA);
    cudaFree(d_wmapB);
    cudaFree(d_gmapA_H);
    cudaFree(d_gmapA_L);
    cudaFree(d_gmapB_H);
    cudaFree(d_gmapB_L);
    cudaFree(d_coeff_H);
    cudaFree(d_coeff_L);
}
