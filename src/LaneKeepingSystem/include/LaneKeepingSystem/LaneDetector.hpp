#ifndef LANE_DETECTOR_HPP_
#define LANE_DETECTOR_HPP_

#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>

/// create your lane detecter
/// Class naming.. it's up to you.
namespace Xycar {
template <typename PREC>
class LaneDetector final
{
public:
    using Ptr = LaneDetector*; /// < Pointer type of the class(it's up to u)

    static inline const cv::Scalar kRed = { 0, 0, 255 };   /// Scalar values of Red
    static inline const cv::Scalar kGreen = { 0, 255, 0 }; /// Scalar values of Green
    static inline const cv::Scalar kBlue = { 255, 0, 0 };  /// Scalar values of Blue

    std::pair<double, double> mresult = { 70, 470 };

    LaneDetector(const YAML::Node& config) { setConfiguration(config); }
    void yourOwnFunction(const cv::Mat img);
    std::pair<double, std::pair<double, double>> Hough(const cv::Mat img);
    cv::Mat regionOfInterest(cv::Mat img);
    std::pair<double, double> calculatePoints(std::pair<double, double> mresult, std::vector<cv::Vec4i> lines);

private:
    int32_t mImageWidth;
    int32_t mImageHeight;
    int32_t mYOffset;

    // Debug Image and flag
    cv::Mat mDebugFrame; /// < The frame for debugging
    void setConfiguration(const YAML::Node& config);
    bool mDebugging;
};
} // namespace Xycar

#endif // LANE_DETECTOR_HPP_