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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include "LaneKeepingSystem/LaneDetector.hpp"

namespace Xycar {

template <typename PREC>
void LaneDetector<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();

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
cv::Mat LaneDetector<PREC>::regionOfInterest(cv::Mat src) {

	cv::Mat src_roi;
	int width = src. cols;
	int height = src.rows;

	const int x = 250;
	const int y_offset = 400;

	const cv::Point p1(0, y_offset - 10), p2(x, y_offset + 10);
	const cv::Point p3(width - x, y_offset - 10), p4(width, y_offset + 10);

	cv::Mat roi_mask = cv::Mat::zeros(height, width, CV_8UC1);
	cv::rectangle(roi_mask, p1, p2, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
	cv::rectangle(roi_mask, p3, p4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
	cv::bitwise_and(src, roi_mask, src_roi);

	return src_roi;
}

template <typename PREC>
std::pair<double, double> LaneDetector<PREC>::calculatePoints(std::pair <double, double> prev_result, std::vector<cv::Vec4i> lines) {
	std::vector<double> results;
	const int width = 640;
	const int y_offset = 400;
	const int sampling_data_size = 10;
    const int pos_threshold = 70;
	int lscnt(0), rscnt(0);
	double mpoint(0);
	std::pair <double, double> cur_result;

	for (cv::Vec4i line : lines) {
		int x1(line[0]);
		int y1(line[1]);
		int x2(line[2]);
		int y2(line[3]);

		double slope = (y2 - y1) / (double)(x2 - x1);
		// (TODO) decide threshold to get rid of outlier 
		// if (slope > l_slope_threshold || slope < r_slope_threshold) {
		// 		continue;
		// }

		double y_intercept = (x2 * y1 - x1 * y2) / (double)(x2 - x1);

		mpoint = (y_offset - y_intercept) / (double)slope;
		results.push_back(mpoint);
	}

	double lpos(0.0), rpos(0.0), lcnt(0.0), rcnt(0.0);
	for (double result : results) {
		if (result <= (width / 2)) {
			lpos += result;
			lcnt++;
		}
		else {
			rpos += result;
			rcnt++;
		}
	}
	cur_result = std::make_pair(lpos / (double)lcnt, rpos / (double)rcnt);

	if (isnan(cur_result.first) == 1){
		cur_result.first = prev_result.first;
	}
	if (isnan(cur_result.second) == 1){
		cur_result.second = prev_result.second;
	}

	if ((abs(prev_result.first - cur_result.first) < pos_threshold) || (abs(prev_result.second - cur_result.second) < pos_threshold)){
		prev_result = cur_result;
	}

	results.clear();

	return prev_result;
}

template <typename PREC>
void LaneDetector<PREC>::Hough(const cv::Mat src)
{
    LaneDetector<PREC> lanedetector;

    std::pair<double, double> result;
	const int y_offset = 400;
    const int width = 640;

    if(!src.empty()){
        cv::imshow("frame", src);
        cv::waitKey(33);   
    }

    cv::Mat src_Gblur, src_edge;
    cvtColor(src, src_Gblur, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(src_Gblur, src_Gblur, cv::Size(3, 3), 0);
    Canny(src_Gblur, src_edge, 240, 250);

    cv::Mat src_roi = lanedetector.regionOfInterest(src_edge);

    std::vector<cv::Vec4i> lines;
    HoughLinesP(src_roi, lines, 1, CV_PI / 180, 20, 15, 20);

    // Draw lines using Hough Transform results
    for (cv::Vec4i l : lines) {
        line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    result = lanedetector.calculatePoints(result, lines);

    // Draw a line and points using calculated results
    line(src, cv::Point(0, y_offset), cv::Point(width, y_offset), cv::Scalar(0, 255, 128), 1, cv::LINE_AA);
    circle(src, cv::Point(result.first, y_offset), 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA, 0);
    circle(src, cv::Point(result.second, y_offset), 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA, 0);
    circle(src, cv::Point((result.first + result.second) / 2, y_offset), 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA, 0);
    circle(src, cv::Point(width / 2, y_offset), 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA, 0);

    imshow("result", src);
}


template class LaneDetector<float>;
template class LaneDetector<double>;
} // namespace Xycar
