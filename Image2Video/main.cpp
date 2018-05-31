#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>

#include <boost/filesystem.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft, vector<string> &strImageRight, vector<double> &vTimestamps)
{
}

bool image2video(const string &pathName, const string &videoName)
{
    // Generate the image file list using boost
    vector<string> filesInPath;
    boost::filesystem::path p(pathName);
    boost::filesystem::directory_iterator it(p);
    boost::filesystem::directory_iterator ed;
    boost::filesystem::directory_entry entry;
    while(it != ed)
    {
        entry = *it++;
        string oss = entry.path().string();
        filesInPath.push_back(oss);
        //ostringstream oss;
        //oss << *it++;
        //filesInPath.push_back(oss.str());

    }

    cout << "Total read " << filesInPath.size() << " images." << endl;
    if (filesInPath.empty())
        return false;

    //for (auto & ele : filesInPath)
        //cout << ele << endl;
    Mat img = imread(filesInPath.at(0));
    //VideoWriter cap(videoName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 24, img.size());
    VideoWriter cap(videoName, VideoWriter::fourcc('P', 'I', 'M', '1'), 24, img.size());
    //VideoWriter cap(videoName, -1, 24, img.size());

    assert(cap.isOpened());
    for (auto & ele : filesInPath)
    {
        cout << ele << endl;
        imshow("video", img);
        if (img.empty())
        {
            cout << "read failed." << endl;
            continue;
        }
        img = imread(ele);
        cap << img;

        waitKey(20);
    }

    cap.release();

    waitKey(0);

    return true;
}

int main(int argc, char **argv) {
    std::cout << "Hello, World!" << std::endl;

    string pathName(argv[1]);
    string videoName = "video.avi";

    if (image2video(pathName, videoName))
        cout << "Video has been saved." << endl;


    return 0;
}
