// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file LaneDetector.cpp
 * @author Jeongmin Kim
 * @author Jeongbin Yim
 * @brief lane detector class source file
 * @version 2.1
 * @date 2023-10-13
 */

#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "LaneKeepingSystem/LaneDetector.hpp"

namespace Xycar {

template <typename PREC>
void LaneDetector<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();
    mMinThres = config["IMAGE"]["MINTHRES"].as<int32_t>();
    mMaxThres = config["IMAGE"]["MAXTHRES"].as<int32_t>();

    mSrcPoint_X0 = config["IMAGE"]["SRCPT_X0"].as<int32_t>();
    mSrcPoint_X1 = config["IMAGE"]["SRCPT_X1"].as<int32_t>();
    mSrcPoint_X2 = config["IMAGE"]["SRCPT_X2"].as<int32_t>();
    mSrcPoint_X3 = config["IMAGE"]["SRCPT_X3"].as<int32_t>();
    mSrcPoint_Y0 = config["IMAGE"]["SRCPT_Y0"].as<int32_t>();
    mSrcPoint_Y1 = config["IMAGE"]["SRCPT_Y1"].as<int32_t>();
    mSrcPoint_Y2 = config["IMAGE"]["SRCPT_Y2"].as<int32_t>();
    mSrcPoint_Y3 = config["IMAGE"]["SRCPT_Y3"].as<int32_t>();

    mDstPoint_X0 = config["IMAGE"]["DSTPT_X0"].as<int32_t>();
    mDstPoint_X1 = config["IMAGE"]["DSTPT_X1"].as<int32_t>();
    mDstPoint_X2 = config["IMAGE"]["DSTPT_X2"].as<int32_t>();
    mDstPoint_X3 = config["IMAGE"]["DSTPT_X3"].as<int32_t>();
    mDstPoint_Y0 = config["IMAGE"]["DSTPT_Y0"].as<int32_t>();
    mDstPoint_Y1 = config["IMAGE"]["DSTPT_Y1"].as<int32_t>();
    mDstPoint_Y2 = config["IMAGE"]["DSTPT_Y2"].as<int32_t>();
    mDstPoint_Y3 = config["IMAGE"]["DSTPT_Y3"].as<int32_t>();

    mGausBlurSigma = config["IMAGE"]["BLURSIGMA"].as<float>();

    std::vector<cv::Point2f> mSrcPts(4), mDstPts(4);
    std::vector<cv::Point> mPts(4), mWarpLeftLine(2), mWarpRightLine(2);

    mSrcPts[0] = cv::Point2f(mSrcPoint_X0, mSrcPoint_Y0); mSrcPts[1] = cv::Point2f(mSrcPoint_X1, mSrcPoint_Y1);
    mSrcPts[2] = cv::Point2f(mSrcPoint_X2, mSrcPoint_Y2); mSrcPts[3] = cv::Point2f(mSrcPoint_X3, mSrcPoint_Y3);

    mDstPts[0] = cv::Point2f(mDstPoint_X0, mDstPoint_Y0); mDstPts[1] = cv::Point2f(mDstPoint_X1, mDstPoint_Y1);
    mDstPts[2] = cv::Point2f(mDstPoint_X2, mDstPoint_Y2); mDstPts[3] = cv::Point2f(mDstPoint_X3, mDstPoint_Y3);

	mPts[0] = cv::Point(mSrcPts[0]); mPts[1] = cv::Point(mSrcPts[1]);
    mPts[2] = cv::Point(mSrcPts[2]); mPts[3] = cv::Point(mSrcPts[3]);

    mPerMatToDst = cv::getPerspectiveTransform(mSrcPts, mDstPts);
    mPerMatToSrc = cv::getPerspectiveTransform(mDstPts, mSrcPts);

    mDebugging = config["DEBUG"].as<bool>();
}

/*
Example Function Form
*/
template <typename PREC>
cv::Mat LaneDetector<PREC>::Vthres(const cv::Mat img)
{
    cv::Mat v_thres = cv::Mat::zeros(mImageWidth, mImageHeight, CV_8UC1);
    cv::warpPerspective(img, mBirdEyeImg, mPerMatToDst, cv::Size(mImageWidth, mImageHeight), cv::INTER_LINEAR);
    cv::cvtColor(mBirdEyeImg, mHsvImg, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_planes;
    cv::split(mBirdEyeImg, hsv_planes);
    cv::Mat v_plane = hsv_planes[2];
    v_plane = 255 - v_plane;

    int means = mean(v_planes)[2];
    v_plane = v_plane + (100 - means);

    //cv::GaussianBlur(v_plane, cv)

        
    return v_thres;
}

template <typename PREC>
void LaneDetector<PREC>::yourOwnFunction(const cv::Mat img)
{
    // write your code.
    // &
    // you must specify your own function to your LaneDetector.hpp file.
    if(!img.empty()){
        cv::imshow("frame", img);
        cv::waitKey(33);   
    }
}


template class LaneDetector<float>;
template class LaneDetector<double>;
} // namespace Xycar
