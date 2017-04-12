#include "SVD.h"

/* Parameters */
//#define M 369
//#define N 369
//#define LDA M
//#define LDU M
//#define LDVT N

void printMaxtrix(int m, int n, const float*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++)
	{
		for (int col = 0; col < n; col++)
		{
			float Areg = A[row + col*lda];
			printf("%s(%d,%d)=%f\n", name, row + 1, col + 1, Areg);
		}
	}
}

int main(int argc, char **argv)
{

	if(argc != 2)
	{
		cout << "no image data input..." << endl;
		return -1;
	}
	Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if(!img.data)
	{
		cout << "read image data failed..." << endl;
		return -1;
	}

	//Mat img_svd;
	img.convertTo(img, CV_32F, 1.0);
	float *A = (float *)img.data;
	int m = img.rows;
	int n = img.cols;
	const int M = m;
	const int N = n;
	const int LDA = M;
	const int LDU = M;
	const int LDVT = N;
	//int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT;
	int lda = LDA;
	int ldu = LDU;
	int ldvt = LDVT;

	//float s[N], u[LDU*M], vt[LDVT*N];
	float *s, *u, *vt;
	s = (float *)malloc(sizeof(float) * N);
	u = (float *)malloc(sizeof(float) * LDU * M);
	vt = (float *)malloc(sizeof(float) * LDVT * N);

	clock_t start, end ;
	double duration;

	printf("A = (matlab base-1)\n");
	//printMaxtrix(m, n, A, lda, "A");
	printf("====\n");

	SVDT svd;
	start = clock();
	svd.SVDcompute(m, n, lda, ldu, ldvt, A, u, s, vt);
	end = clock();
	duration = double((end - start) / CLOCKS_PER_SEC);

	printf("U= (matlab base-1)\n");
	//printMaxtrix(ldu, m, u, ldu, "U");
	printf("====\n");
	printf("S= (matlab base-1)\n");
	//printMaxtrix(n, 1, s, n, "S");
	printf("====\n");
	printf("VT= (matlab base-1)\n");
	//printMaxtrix(ldvt, n, vt, ldvt, "VT");
	printf("====\n");

	printf("duration=%f sec\n", duration);

	//opencv part
	cv::SVD svd1;
	Mat cvA(LDA, N, CV_64F, &A);
	cvA = cvA.t();
	Mat cvw, cvu, cvvt;
	start = clock();
	svd1.compute(cvA, cvw, cvu, cvvt);
	end = clock();
	/*
	cout << "cvA = " << endl << " " << cvA << endl << endl;
	cout << "cvw = " << endl << " " << cvw << endl << endl;
	cout << "cvu = " << endl << " " << cvu << endl << endl;
	cout << "cvvt = " << endl << " " << cvvt << endl << endl;
	*/

	duration = double((end - start) / CLOCKS_PER_SEC);
	printf("duration=%f ms\n", duration * 1000);

	free(s);
	free(u);
	free(vt);

	return 0;
}
