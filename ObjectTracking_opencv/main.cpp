#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    VideoCapture cap;
    cap.open(0);

    if(!cap.isOpened())
    {
        cout << "Cannot open the camera ..." << endl;
        return -1;
    }

    Ptr<Tracker> tracker = Tracker::create("MIL");

    cout << "Width : " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "Height : " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;

    Mat frames;
    // Define an initial bounding box
    Rect2d bbox(160, 80, 280, 280);

    cap >> frames;
    tracker->init(frames, bbox);

    bool stop = false;
    while(!stop)
    {
        cap >> frames;

        // update tracking result
        tracker->update(frames, bbox);

        // Draw bounding box
        rectangle(frames, bbox, Scalar(255, 0, 0), 2, 1);

        // display the result
        imshow("Tracking", frames);
        if(waitKey(33) >= 0)
            stop = true;
    }


    return 0;
}
