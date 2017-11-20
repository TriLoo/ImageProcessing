#include "gff.h"

GFF::GFF(int wid, int hei, int lr, int gr) : WMap(wid, hei, lr, gr)
{
}

GFF::~GFF()
{

}

void GFF::gffFusion(float *d_imgOut, float *d_imgInA, float *d_imgInB, int wid, int hei, int tsr, int lr, int gr,
                    double gsig, int guir, double guieps)
{

}

void GFF::gffFusionTest(float *imgOut, float *imgInA, float *imgInB, int wid, int hei, int tsr, int lr, int gr,
                        double gsig, int guir, double guieps)
{

}

void GFF::Merge(float *d_imgOut, vector<float *> &d_imgInS, vector<float *> &d_wmapInS, const int wid, const int hei)
{

}

void GFF::MergeTest(float *imgOut, vector<float *> &imgInS, vector<float *> &wampInS, int wid, int hei)
{

}