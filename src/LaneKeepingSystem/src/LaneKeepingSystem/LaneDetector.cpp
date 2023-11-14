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
#include <opencv2/core/mat.hpp>
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
    cv::Mat mFrame, mBirdEyeImg, mHsvImg, mGausImg;

    mDebugging = config["DEBUG"].as<bool>();
}

/*
Example Function Form
*/
template <typename PREC>
void LaneDetector<PREC>::totalFunction(const cv::Mat img)
{
    if(img.empty()){
        std::cerr << "Not img" << std:: endl;
    }
    else{
        cv::Mat v_thres = cv::Mat::zeros(mImageWidth, mImageHeight, CV_8UC1);

        cv::warpPerspective(img, mBirdEyeImg, mPerMatToDst, cv::Size(mImageWidth, mImageHeight));
        cv::cvtColor(mBirdEyeImg, mHsvImg, cv::COLOR_BGR2HSV);
        
        std::vector<cv::Mat> hsv_planes;
        cv::split(mBirdEyeImg, hsv_planes);
        cv::Mat v_plane = hsv_planes[2];
        v_plane = 255 - v_plane;

        int means = mean(v_plane)[0];
        v_plane = v_plane + (100 - means);

        cv::GaussianBlur(v_plane, v_plane, cv::Size(), mGausBlurSigma);
        cv::inRange(v_plane, mMinThres, mMaxThres, v_thres);

		cv::arrowedLine(mBirdEyeImg, cv::Point(mImageWidth / 2, mImageHeight), cv::Point(mImageWidth / 2, mImageHeight - 40), cv::Scalar(255, 0, 255), 3);

        int left_l_init = 0, left_r_init = 0;
        int right_l_init = mImageWidth, right_r_init = mImageWidth;

        for(int x = 0; x < mImageWidth; x++){
            if(x < mImageWidth / 2){
                if(v_thres.at<uchar>(mImageHeight - 1, x) == 255 && left_l_init == 0){
                    left_l_init = x;
                    left_r_init = x;
                }
				if (v_thres.at<uchar>(mImageHeight - 1, x) == 255 && left_r_init != 0) {
					left_r_init = x;
				}
			}
			else{
				if (v_thres.at<uchar>(mImageHeight - 1, x) == 255 && right_l_init == 640) {
					right_l_init = x;
					right_r_init = x;
				}
				if (v_thres.at<uchar>(mImageHeight - 1, x) == 255 && right_r_init != 640) {
					right_l_init = x;
				}
			}
        }

        int left_mid_point = (left_l_init + left_r_init) / 2;
        int right_mid_point = (right_l_init + right_r_init) / 2;

        numSlidingWindows(left_mid_point, right_mid_point, mBirdEyeImg, v_thres, mImageWidth, mImageHeight, mPerMatToSrc, img);

        cv::imshow("frame_", img);
        cv::imshow("check", mBirdEyeImg);
        cv::waitKey(33);
    }
}

template <typename PREC>
void LaneDetector<PREC>::numSlidingWindows(const int left_mid, const int right_mid, const cv::Mat roi, const cv::Mat v_thres,
                            const int w, const int h, const cv::Mat per_mat_tosrc, const cv::Mat frame)
{
	// define constant for sliding window
	int n_windows = 12;
	int window_height = static_cast<int>(h / n_windows);
	int window_width = static_cast<int>(w / n_windows * 1.2);
	int margin = window_width >> 1;

    std::vector<cv::Point> l_points(n_windows), r_points(n_windows);
	std::vector<cv::Point> m_points(n_windows);

    int left_mid_point = left_mid;
    int right_mid_point = right_mid;

	int win_y_high = h - window_height;
	int win_y_low = h;
	int win_x_leftb_right = left_mid_point + margin;
	int win_x_leftb_left = left_mid_point - margin;
	int win_x_rightb_right = right_mid_point + margin;
	int win_x_rightb_left = right_mid_point - margin;

	l_points[0] = cv::Point(left_mid_point, static_cast<int>((win_y_high + win_y_low) >> 1));
	r_points[0] = cv::Point(right_mid_point, static_cast<int>((win_y_high + win_y_low) >> 1));
	m_points[0] = cv::Point(static_cast<int>((left_mid_point + right_mid_point) >> 1), static_cast<int>((win_y_high + win_y_low) >> 1));

	// window search start, i drew the init box at the bottom, so i start from 1 to n_windows
	for (int window = 0; window < n_windows; window++) {

		win_y_high = h - (window + 1) * window_height;
		win_y_low = h - window * window_height;
		win_x_leftb_right = left_mid_point + margin;
		win_x_leftb_left = left_mid_point - margin;
		win_x_rightb_right = right_mid_point + margin;
		win_x_rightb_left = right_mid_point - margin;

		int offset = static_cast<int>((win_y_high + win_y_low) >> 1);
		int pixel_thres = window_width * 0.1;
		int ll = 0, lr = 0; int rl = w, rr = w;
		
		int li = 0; // nonzero가 몇개인지 파악하기 위한 벡터에 사용될 인자
		// window의 위치를 고려해서 벡터에 집어넣으면 불필요한 부분이 많아질 수 있다. 어차피 0의 개수를 구하기 위한 벡터이므로 0부터 window_width+1 개수만큼 생성
		std::vector<int> lhigh_vector(window_width + 1); // nonzero가 몇개 인지 파악할 때 사용할 벡터
		for (auto x = win_x_leftb_left; x < win_x_leftb_right; x++) {
			li++;
			lhigh_vector[li] = v_thres.at<uchar>(offset, x);

			// 차선의 중앙을 계산하기 위해 255 시작점과 255 끝점을 계산
			if (v_thres.at<uchar>(offset, x) == 255 && ll == 0) {
				ll = x;
				lr = x;
			}
			if (v_thres.at<uchar>(offset, x) == 255 && lr != 0) {
				lr = x;
			}
		}

		int ri = 0;// nonzero가 몇개인지 파악하기 위한 벡터에 사용될 인자
		// window의 위치를 고려해서 벡터에 집어넣으면 불필요한 부분이 많아질 수 있다. 어차피 0의 개수를 구하기 위한 벡터이므로 0부터 window_width+1 개수만큼 생성
		std::vector<int> rhigh_vector(window_width + 1);
		for (auto x = win_x_rightb_left; x < win_x_rightb_right; x++) {
			ri++;
			rhigh_vector[ri] = v_thres.at<uchar>(offset, x);
			if (v_thres.at<uchar>(offset, x) == 255 && rl == w) {
				rl = x;
				rr = x;
			}
			if (v_thres.at<uchar>(offset, x) == 255 && lr != w) {
				rr = x;
			}
		}

		// window안에서 0이 아닌 픽셀의 개수를 구함
		int lnonzero = cv::countNonZero(lhigh_vector);
		int rnonzero = cv::countNonZero(rhigh_vector);

		// 방금 구했던 255 픽셀 시작 지점과 끝 지점의 중앙 값을 다음 sliding_window의 중앙으로 잡는다.
		if (lnonzero >= pixel_thres) {
			left_mid_point = (ll + lr) >> 1;
		}
		if (rnonzero >= pixel_thres) {
			right_mid_point = (rl + rr) >> 1;
		}

		// 차선 중앙과 탐지한 차선과의 거리 측정
		int lane_mid = (right_mid_point + left_mid_point) >> 1;
		int left_diff = lane_mid - left_mid_point;
		int right_diff = -(lane_mid - right_mid_point);

#if 1
		// 한쪽 차선의 nonzero가 임계값을 넘지 못할 경우 중간을 기점으로 반대편 차선 위치를 기준으로 대칭
		if (lnonzero < pixel_thres && rnonzero > pixel_thres) {
			lane_mid = right_mid_point - right_diff;
			left_mid_point = lane_mid - right_diff;
		}
		else if (lnonzero > pixel_thres && rnonzero < pixel_thres) {
			lane_mid = left_mid_point + left_diff;
			right_mid_point = lane_mid + left_diff;
		}
#else
		// 지난 프레임에서의 픽셀값을 기억하고 nonzero가 임계값을 넘지 못할 경우 지난 프레임의 해당 윈도우 번호의 값을 불러옴
		if (lnonzero < pixel_thres && rnonzero > pixel_thres) {
			left_mid_point = l_points[window].x;
			lane_mid = (right_mid_point + left_mid_point) / 2;
		}
		else if (lnonzero > pixel_thres && rnonzero < pixel_thres && r_points[window].x != 0) {
			right_mid_point = r_points[window].x;
			lane_mid = (right_mid_point + left_mid_point) / 2;
		}

#endif
		// draw window at v_thres
		rectangle(roi, cv::Rect(win_x_leftb_left, win_y_high, window_width, window_height), cv::Scalar(0, 150, 0), 2);
		rectangle(roi, cv::Rect(win_x_rightb_left, win_y_high, window_width, window_height), cv::Scalar(150, 0, 0), 2);


		m_points[window] = cv::Point(lane_mid, static_cast<int>((win_y_high + win_y_low) >> 1));
		l_points[window] = cv::Point(left_mid_point, static_cast<int>((win_y_high + win_y_low) >> 1));
		r_points[window] = cv::Point(right_mid_point, static_cast<int>((win_y_high + win_y_low) >> 1));
	}

	cv::Vec4f left_line, right_line, mid_line;
	
	cv::fitLine(l_points, left_line, cv::DIST_L2, 0, 0.01, 0.01); // 출력의 0,1 번째 인자는 단위벡터, 3,4번째 인자는 선 위의 한 점
	cv::fitLine(r_points, right_line, cv::DIST_L2, 0, 0.01, 0.01);
	cv::fitLine(m_points, mid_line, cv::DIST_L2, 0, 0.01, 0.01);

	// 방향이 항상 아래를 향하도록 만들기 위해 단위 벡터의 방향을 바꿔준다.
	if (left_line[1] > 0) {
		left_line[1] = -left_line[1];
	}
	if (right_line[1] > 0) {
		right_line[1] = right_line[1];
	}
	if (mid_line[1] > 0) {
		mid_line[1] = mid_line[1];
	}

	int lx0 = left_line[2], ly0 = left_line[3]; // 선 위의 한 점
	int lx1 = lx0 + h / 2 * left_line[0], ly1 = ly0 + h / 2 * left_line[1]; // 단위 벡터 -> 그리고자 하는 길이를 빼주거나 더해줌
	int lx2 = 2 * lx0 - lx1, ly2 = 2 * ly0 - ly1;

	int rx0 = right_line[2], ry0 = right_line[3];
	int rx1 = rx0 + h / 2 * right_line[0], ry1 = ry0 + h / 2 * right_line[1];
	int rx2 = 2 * rx0 - rx1, ry2 = 2 * ry0 - ry1;

	int mx0 = mid_line[2], my0 = mid_line[3];
	int mx1 = mx0 + h / 2 * mid_line[0], my1 = my0 + h / 2 * mid_line[1];
	int mx2 = 2 * mx0 - mx1, my2 = 2 * my0 - my1;

	line(roi, cv::Point(lx1, ly1), cv::Point(lx2, ly2), cv::Scalar(0, 100, 200), 3);
	line(roi, cv::Point(rx1, ry1), cv::Point(rx2, ry2), cv::Scalar(0, 100, 200), 3);
	line(roi, cv::Point(mx1, my1), cv::Point(mx2, my2), cv::Scalar(0, 0, 255), 3);

	std::vector<cv::Point> warp_left_line(2), warp_right_line(2), pos;
    matrix_oper_pos(frame, mPerMatToSrc, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2);

	return;
}

template <typename PREC>
void LaneDetector<PREC>::matrix_oper_pos(cv::Mat frame, cv::Mat per_mat_tosrc, int lx1, int ly1, int lx2, int ly2, int rx1, int ry1, int rx2, int ry2) {
	std::vector<cv::Point> warp_left_line, warp_right_line;

	int new_lx1, new_ly1, new_lx2, new_ly2;
	new_lx1 = (per_mat_tosrc.at<double>(0, 0) * lx1 + per_mat_tosrc.at<double>(0, 1) * ly1 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx1 + per_mat_tosrc.at<double>(2, 1) * ly1 + per_mat_tosrc.at<double>(2, 2));

	new_ly1 = (per_mat_tosrc.at<double>(1, 0) * lx1 + per_mat_tosrc.at<double>(1, 1) * ly1 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx1 + per_mat_tosrc.at<double>(2, 1) * ly1 + per_mat_tosrc.at<double>(2, 2));

	new_lx2 = (per_mat_tosrc.at<double>(0, 0) * lx2 + per_mat_tosrc.at<double>(0, 1) * ly2 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx2 + per_mat_tosrc.at<double>(2, 1) * ly2 + per_mat_tosrc.at<double>(2, 2));

	new_ly2 = (per_mat_tosrc.at<double>(1, 0) * lx2 + per_mat_tosrc.at<double>(1, 1) * ly2 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx2 + per_mat_tosrc.at<double>(2, 1) * ly2 + per_mat_tosrc.at<double>(2, 2));

	int new_rx1, new_ry1, new_rx2, new_ry2;
	new_rx1 = (per_mat_tosrc.at<double>(0, 0) * rx1 + per_mat_tosrc.at<double>(0, 1) * ry1 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx1 + per_mat_tosrc.at<double>(2, 1) * ry1 + per_mat_tosrc.at<double>(2, 2));

	new_ry1 = (per_mat_tosrc.at<double>(1, 0) * rx1 + per_mat_tosrc.at<double>(1, 1) * ry1 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx1 + per_mat_tosrc.at<double>(2, 1) * ry1 + per_mat_tosrc.at<double>(2, 2));

	new_rx2 = (per_mat_tosrc.at<double>(0, 0) * rx2 + per_mat_tosrc.at<double>(0, 1) * ry2 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx2 + per_mat_tosrc.at<double>(2, 1) * ry2 + per_mat_tosrc.at<double>(2, 2));

	new_ry2 = (per_mat_tosrc.at<double>(1, 0) * rx2 + per_mat_tosrc.at<double>(1, 1) * ry2 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx2 + per_mat_tosrc.at<double>(2, 1) * ry2 + per_mat_tosrc.at<double>(2, 2));

	warp_left_line.push_back(cv::Point(new_lx1, new_ly1)); warp_left_line.push_back(cv::Point(new_lx2, new_ly2));
	warp_right_line.push_back(cv::Point(new_rx1, new_ry1)); warp_right_line.push_back(cv::Point(new_rx2, new_ry2));


	line(frame, cv::Point(new_lx1, new_ly1), cv::Point(new_lx2, new_ly2), cv::Scalar(0, 255, 255), 2);
	line(frame, cv::Point(new_rx1, new_ry1), cv::Point(new_rx2, new_ry2), cv::Scalar(0, 255, 255), 2);

	int offset = 400;
	int lpos = int((offset - warp_left_line[0].y) * ((warp_left_line[1].x - warp_left_line[0].x) / (warp_left_line[1].y - warp_left_line[0].y)) + warp_left_line[0].x);
	int rpos = int((offset - warp_right_line[0].y) * ((warp_right_line[1].x - warp_right_line[0].x) / (warp_right_line[1].y - warp_right_line[0].y)) + warp_right_line[0].x);
	std::vector<cv::Point> pos;
	pos.push_back(cv::Point(lpos, rpos));

	return;
}

template class LaneDetector<float>;
template class LaneDetector<double>;
} // namespace Xycar
