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

#include "LaneKeepingSystem/LaneDetector.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

namespace Xycar {

template <typename PREC>
void LaneDetector<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();
    mYOffset = config["IMAGE"]["Y_OFFSET"].as<int32_t>();

    /**
     * If you want to add your parameter
     * declare your parameter.
     * */

    mDebugging = config["DEBUG"].as<bool>();
}

/*
Example Function Form
*/
template <typename PREC>
cv::Mat LaneDetector<PREC>::regionOfInterest(cv::Mat src)
{
    cv::Mat src_roi;

    const int x = 300;

    const cv::Point p1(0, mYOffset - 10), p2(x, mYOffset + 10);
    const cv::Point p3(mImageWidth - x, mYOffset - 10), p4(mImageWidth, mYOffset + 10);

    cv::Mat roi_mask = cv::Mat::zeros(mImageHeight, mImageWidth, CV_8UC1);
    cv::rectangle(roi_mask, p1, p2, kBlue, -1, cv::LINE_AA);
    cv::rectangle(roi_mask, p3, p4, kBlue, -1, cv::LINE_AA);
    cv::bitwise_and(src, roi_mask, src_roi);

    return src_roi;
}

template <typename PREC>
std::pair<double, double> LaneDetector<PREC>::calculatePoints(std::pair<double, double> prev_result, std::vector<cv::Vec4i> lines)
{
    std::vector<double> results;
    std::pair<double, double> cur_result;
    const int pos_threshold = 50;
    double mpoint(0);

    for (cv::Vec4i line : lines)
    {
        int x1(line[0]);
        int y1(line[1]);
        int x2(line[2]);
        int y2(line[3]);

        double slope = (y2 - y1) / (double)(x2 - x1);
        double y_intercept = (x2 * y1 - x1 * y2) / (double)(x2 - x1);
        // (TODO) decide threshold to get rid of outlier
        // if (slope > l_slope_threshold || slope < r_slope_threshold) {
        // 		continue;
        // }

        mpoint = (mYOffset - y_intercept) / (double)slope;
        results.push_back(mpoint);
    }

    double lpos(0.0), rpos(0.0), lcnt(0.0), rcnt(0.0);
    for (double result : results)
    {
        if (result <= (mImageWidth / 2))
        {
            lpos += result;
            lcnt++;
        }
        else
        {
            rpos += result;
            rcnt++;
        }
    }
    cur_result = std::make_pair(lpos / (double)lcnt, rpos / (double)rcnt);

    if (isnan(cur_result.first) == 1)
    {
        // cur_result.first = prev_result.first;
        cur_result.first = 0;
    }
    if (isnan(cur_result.second) == 1)
    {
        // cur_result.second = prev_result.second;
        cur_result.second = mImageWidth;
    }

    if ((abs(prev_result.first - cur_result.first) < pos_threshold) || (abs(prev_result.second - cur_result.second) < pos_threshold))
    {
        std::cout << "lpos diff : " << abs(prev_result.first - cur_result.first) << "\n";
        std::cout << "rpos diff : " << abs(prev_result.second - cur_result.second) << "\n";
        prev_result = cur_result;
    }

    results.clear();

    return prev_result;
}

template <typename PREC>
std::pair<double, std::pair<double, double>> LaneDetector<PREC>::Hough(const cv::Mat src, std::pair<double, double> prev_result)
{
    if (src.empty()) {}
    else
    {
        cv::Mat src_Gblur, src_edge;
        cvtColor(src, src_Gblur, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(src_Gblur, src_Gblur, cv::Size(3, 3), 0);
        Canny(src_Gblur, src_edge, 240, 250);

        cv::Mat src_roi = regionOfInterest(src_edge);

        std::vector<cv::Vec4i> lines;
        HoughLinesP(src_roi, lines, 1, CV_PI / 180, 20, 15, 20);

        // Draw lines using Hough Transform results
        for (cv::Vec4i l : lines)
        {
            line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), kRed, 2, cv::LINE_AA);
        }

        mresult = calculatePoints(prev_result, lines);
        // std::cout << "result : " << mresult.first << ", " << mresult.second << "\n";

        // Draw a line and points using calculated results
        line(src, cv::Point(0, mYOffset), cv::Point(mImageWidth, mYOffset), kRed, 1, cv::LINE_AA);
        circle(src, cv::Point(mresult.first, mYOffset), 3, kBlue, -1, cv::LINE_AA, 0);
        circle(src, cv::Point(mresult.second, mYOffset), 3, kBlue, -1, cv::LINE_AA, 0);
        circle(src, cv::Point((mresult.first + mresult.second) / 2, mYOffset), 3, kRed, -1, cv::LINE_AA, 0);
        circle(src, cv::Point(mImageWidth / 2, mYOffset), 3, kGreen, -1, cv::LINE_AA, 0);

        double pos_diff = ((mresult.first + mresult.second) / 2) - (mImageWidth / 2);

        int fourcc = cv::VideoWriter::fourcc('D', 'I', 'V', 'X');
        bool isColor = true;

        cv::VideoWriter outputVideo("../../output.avi", fourcc, 33, cv::Size(mImageWidth, mImageHeight), isColor);

        imshow("roi result", src_roi);
        imshow("result", src);
        cv::waitKey(30);

        return std::pair(pos_diff, mresult);
    }
}

template class LaneDetector<float>;
template class LaneDetector<double>;
} // namespace Xycar
