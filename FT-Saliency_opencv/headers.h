//
// Created by smher on 17-12-22.
//

#ifndef SALIENCYDETECTION_HEADERS_H
#define SALIENCYDETECTION_HEADERS_H

#include "iostream"
#include "vector"
#include "stdexcept"
#include "cassert"
#include "chrono"
#include "string"
#include "ctime"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "boost/thread.hpp"
#include "boost/date_time.hpp"
#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
//#include "boost/date_time"

#define LOG(text) __LOG("Saliency.txt", text)

inline void __LOG(const std::string& fname, const std::string &logtext)
{
    std::ofstream fs(fname, std::ostream::app);
    //boost::posix_time::ptime pt = boost::date_time::second_clock::local_time();
    //fs << pt << logtext;
    time_t tt = time(NULL);
    tm *t = localtime(&tt);
    //fs << t->tm_year + 1900 << std::endl;
    fs << "--" ;
    fs << logtext << std::endl;
    fs.close();
}

#endif //SALIENCYDETECTION_HEADERS_H
