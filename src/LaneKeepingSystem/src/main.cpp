#include "LaneKeepingSystem/LaneKeepingSystem.hpp"

using PREC = float;

int32_t main(int32_t argc, char** argv)
{
    ros::init(argc, argv, "Lane Keeping System");
    Xycar::LaneKeepingSystem<PREC> laneKeepingSystem;
    std::pair<double, double> prev_result;
    laneKeepingSystem.run(prev_result);

    return 0;
}
