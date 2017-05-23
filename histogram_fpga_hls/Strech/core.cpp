#include "core.h"
void doHistStrech(hls::stream<uint_8_side_channel> &inStream, hls::stream<uint_8_side_channel> &outStream, unsigned char xMin, unsigned char xMax)
{
#pragma HLS INTERFACE axis port=inStream
#pragma HLS INTERFACE axis port=outStream
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=xMin bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=xMax bundle=CRTL_BUS

	// Calculate and cache the result of xMax-xMin
	float xMax_minus_xMin = xMax-xMin;

	// Iterate on a stream of (320*240)
	for (int idxPixel = 0; idxPixel < (320*240); idxPixel++)
	{
#pragma HLS PIPELINE
		// Read and cache (Block here if FIFO sender is empty)
		uint_8_side_channel currPixelSideChannel = inStream.read();

		uint_8_side_channel dataOutSideChannel;

		// Get the pixel data
		unsigned char x_t = currPixelSideChannel.data;

		// Calculate the histogram strech (calculate in float then convert to 8-bit)
		float y_t_float = ((x_t - xMin) / (xMax_minus_xMin))*255;

		unsigned char y_t = y_t_float;

		// Put data on output stream (side-channel(tlast) way...)
		dataOutSideChannel.data = y_t;
		dataOutSideChannel.keep = currPixelSideChannel.keep;
		dataOutSideChannel.strb = currPixelSideChannel.strb;
		dataOutSideChannel.user = currPixelSideChannel.user;
		dataOutSideChannel.last = currPixelSideChannel.last;
		dataOutSideChannel.id = currPixelSideChannel.id;
		dataOutSideChannel.dest = currPixelSideChannel.dest;

		// Send to the stream (Block if the FIFO receiver is full)
		outStream.write(dataOutSideChannel);
	}
}
