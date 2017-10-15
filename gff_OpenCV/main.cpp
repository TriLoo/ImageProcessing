//#include <iostream>
#include "GffFusion.h"
#include "chrono"

//#define VideoDir "/home/smher/myOpenCV/GffFusion/1.avi"

int main(int argc, char **argv)
{
    //string nameA = "/home/smher/myOpenCV/GffFusion/garage1.jpg";
	//string nameB = "/home/smher/myOpenCV/GffFusion/garage5.jpg";
	string nameA = "/home/smher/myOpenCV/GffFusion/Visual.jpg";
	string nameB = "/home/smher/myOpenCV/GffFusion/Infrared.jpg";
	//string nameA = "/home/smher/myOpenCV/GffFusion/source19_1.tif";
	//string nameB = "/home/smher/myOpenCV/GffFusion/source19_2.tif";
	Mat imgA = imread(nameA, IMREAD_ANYCOLOR);
	Mat imgB = imread(nameB, IMREAD_ANYCOLOR);
	if(!imgA.data)
	{
		cerr << "Read image A fail ..." << endl;
		return -1;
	}
	if(!imgB.data)
	{
		cerr << "Read image B fail ..." << endl;
		return -1;
	}
    Mat Out = Mat::zeros(imgA.size(), CV_32FC3);

    imgA.convertTo(imgA, CV_32FC3, 1.0);
    imgB.convertTo(imgB, CV_32FC3, 1.0);

    GffFusion gf;

    gf.gffFusionColor(imgA, imgB, Out);

    auto start = chrono::steady_clock::now();
	gf.gffFusionColor(imgA, imgB, Out);
    auto stop = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << "Gff Fusion based on OpenCV : " << time_used.count() * 1000.0 << " ms" << endl;

    Out.convertTo(Out, CV_8UC3, 1.0);

    imwrite("Result.jpg", Out);
    imshow("Fusion Result", Out);

    // test the color depth
    /*
    cout << "CV_32F = " << CV_32F << endl;
    cout << "CV_32FC3 = " << CV_32FC3 << endl;
    cout << "CV_8UC1 = " << CV_8UC1 << endl;
    cout << "CV_8UC3 = " << CV_8UC3 << endl;
    */
    //cout << "CV_32F = " << CV_32F << endl;

	waitKey(0);

	return 0;
}

