#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace std;

#define PI 3.14159
#define RAD 2
#define SIG 1;

// Gauss filter
class GF
{
	public:
		//GF(int r):rad(r){}
		GF() = default;
		~GF() = default;

		void createfilter(float *filter,  int rad, float sigma);
		void gaussfilter(float *imgOut, float *imgIn, float *filter, int row, int col, int rad);
	private:
		//int rad = 0;
};

void GF::createfilter(float *filter, int rad, float sigma)
{
	float s = 2 * sigma * sigma;
	int wid = 2 * rad + 1;
	float sum = 0;
	for(int i = -rad; i <= rad; ++i) // row
	{
		for(int j = -rad; j <= rad; ++j)   // col
		{
			float r = i * i + j * j;
			float val = exp(-r/s)/(PI * s);
			sum += val;
			int offset = (i + rad) * wid + (j + rad);
			filter[offset] = val;
		}
	}

	for(int i = -rad; i <= rad; ++i)
		for(int j = -rad; j <= rad; ++j)
		{
			int offset = (i + rad) * wid + (j + rad);
			filter[offset] /= sum;
		}
}

int main(int argc, char **argv)
{
	int rad = RAD;
	float sigma = SIG;
	int width = 2 * rad + 1; 

	float *filter = new float [width * width];

	GF gf;
	gf.createfilter(filter, rad, sigma);

	// print the filter
	for(int i = -rad; i <= rad; ++i)
	{
		for(int j = -rad; j <= rad; ++j)
		{
			int offset = (i + rad) * width + (j + rad);
			cout << filter[offset] << "; ";
			//cout << "Offset = " << offset << "; val = " << filter[offset] << endl;
		}
		cout << endl;
	}


	return 0;
}

