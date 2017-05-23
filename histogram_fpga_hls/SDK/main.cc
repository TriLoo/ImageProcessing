/*
 * Empty C++ Application
 */
#include <stdio.h>
#include "xaxidma.h"
#include "xdohist.h"
#include "xdohiststrech.h"
#include "LenaOnCode.h"
#include "AxiTimerHelper.h"

#define SIZE_ARR (320 * 240)

// Memory used by DMA
#define MEM_BASE_ADDR 0x01000000
#define TX_BUFFER_BASE (MEM_BASE_ADDR + 0x00100000)
#define RX_BUFFER_BASE (MEM_BASE_ADDR + 0x00300000)

// The pointers are for 8-bit memory but their addresses are 32bit
unsigned char *m_dma_buffer_TX = (unsigned char *) TX_BUFFER_BASE;
unsigned char *m_dma_buffer_RX = (unsigned char *) RX_BUFFER_BASE;

unsigned char hist_sw[256];
unsigned char imgOut[SIZE_ARR];
unsigned char imgIn_HW[SIZE_ARR];

// Address of BRAM controller in memory
unsigned int *hist_hw = (unsigned int *)0x40000000;

XAxiDma axiDma;

int initDMA()
{
	XAxiDma_Config *CfgPtr;
	CfgPtr = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
	XAxiDma_CfgInitialize(&axiDma, CfgPtr);

	XAxiDma_IntrDisable(&axiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	return XST_SUCCESS;
}

XDohist doHist;
XDohiststrech doHistStrech;

int initDoHist()
{
	int status;

	XDohist_Config *doHist_cfg;
	doHist_cfg = XDohist_LookupConfig(XPAR_DOHIST_0_DEVICE_ID);
	if(!doHist_cfg)
	{
		printf("Error loading config for doHist_cfg \n");
	}
	status = XDohist_CfgInitialize(&doHist, doHist_cfg);
	if(status != XST_SUCCESS)
	{
		printf("Error initializing for doHist \n");
	}

	return status;
}

int initDoHistStrech()
{
	int status;

	XDohiststrech_Config *doHistStrech_cfg;
	doHistStrech_cfg = XDohiststrech_LookupConfig(XPAR_DOHISTSTRECH_0_DEVICE_ID);
	if(!doHistStrech_cfg)
	{
		printf("Error loading config for doHistStrech_cfg\n");
	}
	status = XDohiststrech_CfgInitialize(&doHistStrech, doHistStrech_cfg);
	if(status != XST_SUCCESS)
	{
		printf("Error initializing for doHistStrech\n");
	}

	return status;
}

void doHistSW(unsigned char *img, unsigned char *hist)
{
	// reset histogram
	for(int idx=0; idx < 256; idx++)
	{
		hist[idx] = 0;
	}
	// calculate the histogram
	for(int idxImg = 0; idxImg < (320 * 240); idxImg++)
	{
		hist[img[idxImg]] = hist[img[idxImg]] + 1;
	}
}

void doHistStrechSW(unsigned char *imgIn, unsigned char *imgOut, unsigned char xMin, unsigned char xMax)
{
	float xMax_minus_xMin = xMax - xMin;
	for(int idxImg = 0; idxImg < (320 * 240); idxImg++)
	{
		// calcualte the histogram strech
		float y_t_float = ((imgIn[idxImg] - xMin) / xMax_minus_xMin) * 255;
		imgOut[idxImg] = y_t_float;
	}
}

int main()
{
	initDMA();
	initDoHist();
	initDoHistStrech();
	AxiTimerHelper axiTimer;

	printf("Doing histogram on SW\n");
	axiTimer.startTimer();
	doHistSW(img, hist_sw);
	axiTimer.stopTimer();

	double hist_SW_elapsed = axiTimer.getElapsedTimerInSeconds();
	printf("Histogram SW execution time: %f sec\n", hist_SW_elapsed);

	// Get min value
	unsigned char xMin;
	for(int idxMin = 0; idxMin < 256; idxMin++)
	{
		xMin = idxMin;
		if(hist_sw[idxMin])
			break;
	}

	// Get max value
	unsigned char xMax;
	for(int idxMax = 255; idxMax >= 0; idxMax--)
	{
		xMax = idxMax;
		if(hist_sw[idxMax])
			break;
	}

	printf("(SW) xMin = %d, xMax = %d\n", xMin, xMax);

	printf("Doing histogram strech SW \n");
	axiTimer.startTimer();
	doHistStrechSW(img, imgOut, xMin, xMax);
	axiTimer.stopTimer();
	// sum the elapsed time
	double histStrech_SW_elapsed = axiTimer.getElapsedTimerInSeconds();
	printf("Histogram strech SW execution time: %f sec\n", histStrech_SW_elapsed);

	// Now doing on hardware
	// populate data(get image from header and put on memory)
	for(int idx = 0; idx < SIZE_ARR; idx++)
	{
		imgIn_HW[idx] = img[idx];
	}

	XDohist_Start(&doHist);
	// at this point, we don;t care on the result of the strech algorithm because we're still getting the histogram
	XDohiststrech_Set_xMax(&doHistStrech, 255);
	XDohiststrech_Set_xMin(&doHistStrech, 0);
	XDohiststrech_Start(&doHistStrech);

	printf("Test 1\n");

	axiTimer.startTimer();
	Xil_DCacheFlushRange((u32)imgIn_HW, SIZE_ARR*sizeof(unsigned char));
	Xil_DCacheFlushRange((u32)m_dma_buffer_RX, SIZE_ARR*sizeof(unsigned char));

	XAxiDma_SimpleTransfer(&axiDma,(u32)imgIn_HW,SIZE_ARR*sizeof(unsigned char),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDma,(u32)m_dma_buffer_RX,SIZE_ARR*sizeof(unsigned char),XAXIDMA_DEVICE_TO_DMA);

	printf("Test 2\n");

	//Wait transfers to finish
	while(XAxiDma_Busy(&axiDma,XAXIDMA_DMA_TO_DEVICE));
	while(XAxiDma_Busy(&axiDma,XAXIDMA_DEVICE_TO_DMA));

	// Invalidate the cache to avoid reading garbage
	Xil_DCacheInvalidateRange((u32)m_dma_buffer_RX, SIZE_ARR * sizeof(unsigned char));
	axiTimer.stopTimer();
	double hist_HW_elapsed = axiTimer.getElapsedTimerInSeconds();
	printf("Histogram HW execution time: %f sec\n", hist_HW_elapsed);

	// Get min Value

	for(int idxMin = 0; idxMin < 256; idxMin++)
	{
		xMin = idxMin;
		if(hist_hw[idxMin])
			break;
	}
	// get max value
	for(int idxMax = 255; idxMax >= 0; idxMax--)
	{
		xMax = idxMax;
		if(hist_hw[idxMax])
			break;
	}

	printf("(HW) xMin = %d, xMax = %d\n", xMin, xMax);

	// Now do the histogram strech
	XDohist_Start(&doHist);
	XDohiststrech_Set_xMax(&doHistStrech, xMax);
	XDohiststrech_Set_xMin(&doHistStrech, xMin);
	XDohiststrech_Start(&doHistStrech);

	axiTimer.startTimer();
	Xil_DCacheFlushRange((u32)imgIn_HW, SIZE_ARR * sizeof(unsigned char));
	Xil_DCacheFlushRange((u32)m_dma_buffer_RX, SIZE_ARR * sizeof(unsigned char));

	XAxiDma_SimpleTransfer(&axiDma, (u32)imgIn_HW, SIZE_ARR * sizeof(unsigned char), XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&axiDma, (u32)m_dma_buffer_RX, SIZE_ARR * sizeof(unsigned char), XAXIDMA_DEVICE_TO_DMA);

	// Wait transfers to finish
	while(XAxiDma_Busy(&axiDma, XAXIDMA_DMA_TO_DEVICE));
	while(XAxiDma_Busy(&axiDma, XAXIDMA_DEVICE_TO_DMA));

	// Invalidate the cache to avoid reading garbage
	Xil_DCacheInvalidateRange((u32)m_dma_buffer_RX, SIZE_ARR * sizeof(unsigned char));
	axiTimer.stopTimer();
	double histStrech_HW_elapsed = axiTimer.getElapsedTimerInSeconds();
	printf("Histogram strech HW execution time : %f sec\n", histStrech_HW_elapsed);

	printf("DMA out address : 0x%X\n", m_dma_buffer_RX);

	int imgMistmatch = 0;

	for(int idxComp = 0; idxComp < SIZE_ARR; idxComp++)
	{
		if(imgOut[idxComp] != m_dma_buffer_RX[idxComp])
		{
			printf("Invalid response\n");
			imgMistmatch = 1;
		}
	}

	if(!imgMistmatch)
		printf("SW and HW images is the same\n");
	else
		printf("SW and HW images is not the same\n");

	return 0;
}
