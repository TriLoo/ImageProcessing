#include "core.h"
void doHist(hls::stream<uint_8_side_channel> &inStream, int histo[256])
{
#pragma HLS INTERFACE axis port=inStream
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
#pragma HLS INTERFACE bram port=histo

	// Initialize always before calculating to zero
	for (int idxHist = 0; idxHist < 256; idxHist++)
	{
//#pragma HLS PIPELINE
		histo[idxHist] = 0;
	}

	// Iterate on a stream of (320*240)
	for (int idxPixel = 0; idxPixel < (320*240); idxPixel++)
	{
//#pragma HLS PIPELINE
		// Read and cache (Block here if FIFO sender is empty)
		uint_8_side_channel currPixelSideChannel = inStream.read();

		// Calculate the histogram
		histo[currPixelSideChannel.data]+=1;
	}

}
