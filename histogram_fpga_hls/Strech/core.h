// Use the class stream
#include <hls_stream.h>

// Use the axi stream side-channel (TLAST,TKEEP,TUSR,TID)
#include <ap_axi_sdata.h>
typedef ap_axiu<8,2,5,6> uint_8_side_channel;

void doHistStrech(hls::stream<uint_8_side_channel> &inStream, hls::stream<uint_8_side_channel> &outStream, unsigned char xMin, unsigned char xMax);
