// Use the class stream
#include <hls_stream.h>

// Use the axi stream side-channel (TLAST,TKEEP,TUSR,TID)
#include <ap_axi_sdata.h>
typedef ap_axiu<8,2,5,6> uint_8_side_channel;

void doHist(hls::stream<uint_8_side_channel> &inStream, int histo[256]);
