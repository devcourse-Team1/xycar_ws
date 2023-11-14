#ifndef LANE_DETECTOR_HPP_
#define LANE_DETECTOR_HPP_

#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>
#include <vector>

/// create your lane detecter
/// Class naming.. it's up to you.
namespace Xycar {
template <typename PREC>
class LaneDetector final
{
public:
    using Ptr = LaneDetector*; /// < Pointer type of the class(it's up to u)

    static inline const cv::Scalar kRed = {0, 0, 255}; /// Scalar values of Red
    static inline const cv::Scalar kGreen = {0, 255, 0}; /// Scalar values of Green
    static inline const cv::Scalar kBlue = {255, 0, 0}; /// Scalar values of Blue

    LaneDetector(const YAML::Node& config) {setConfiguration(config);}
    void totalFunction(const cv::Mat img);
    void yourOwnFunction(const cv::Mat img);


private:
    int32_t mImageWidth;
    int32_t mImageHeight;
    int32_t mMinThres;
    int32_t mMaxThres;
    
    int32_t mSrcPoint_X0;
    int32_t mSrcPoint_X1;
    int32_t mSrcPoint_X2;
    int32_t mSrcPoint_X3;
    int32_t mSrcPoint_Y0;
    int32_t mSrcPoint_Y1;
    int32_t mSrcPoint_Y2;
    int32_t mSrcPoint_Y3;
    
    int32_t mDstPoint_X0;
    int32_t mDstPoint_X1;
    int32_t mDstPoint_X2;
    int32_t mDstPoint_X3;
    int32_t mDstPoint_Y0;
    int32_t mDstPoint_Y1;
    int32_t mDstPoint_Y2;
    int32_t mDstPoint_Y3;

    float mGausBlurSigma;

    cv::Mat mBirdEyeImg, mHsvImg, mGausImg;
    cv::Mat mPerMatToDst, mPerMatToSrc;
    
    std::vector<cv::Point2f> mSrcPts, mDstPts;
    std::vector<cv::Point> mPts, mWarpLeftLine, mWarpRightLine;
    // Debug Image and flag
    cv::Mat mDebugFrame; /// < The frame for debugging
    void setConfiguration(const YAML::Node& config);
    bool mDebugging;
};
}

#endif // LANE_DETECTOR_HPP_