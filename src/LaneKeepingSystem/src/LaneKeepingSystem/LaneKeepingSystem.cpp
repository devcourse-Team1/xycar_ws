#include "LaneKeepingSystem/LaneKeepingSystem.hpp"

namespace Xycar {
template <typename PREC>
LaneKeepingSystem<PREC>::LaneKeepingSystem()
{
    std::string configPath;
    mNodeHandler.getParam("config_path", configPath);
    YAML::Node config = YAML::LoadFile(configPath);

    mPID = new PIDController<PREC>(config["PID"]["P_GAIN"].as<PREC>(), config["PID"]["I_GAIN"].as<PREC>(), config["PID"]["D_GAIN"].as<PREC>());
    mPID_CURVE = new PIDController<PREC>(config["PID"]["P_GAIN_CURVE"].as<PREC>(), config["PID"]["I_GAIN_CURVE"].as<PREC>(), config["PID"]["D_GAIN_CURVE"].as<PREC>());
    mMovingAverage = new MovingAverageFilter<PREC>(config["MOVING_AVERAGE_FILTER"]["SAMPLE_SIZE"].as<uint32_t>());
    mLaneDetector = new LaneDetector<PREC>(config);
    /*
        create your lane detector.
    */
    setParams(config);

    mPublisher = mNodeHandler.advertise<xycar_msgs::xycar_motor>(mPublishingTopicName, mQueueSize);
    mSubscriber = mNodeHandler.subscribe(mSubscribedTopicName, mQueueSize, &LaneKeepingSystem::imageCallback, this);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::setParams(const YAML::Node& config)
{
    mPublishingTopicName = config["TOPIC"]["PUB_NAME"].as<std::string>();
    mSubscribedTopicName = config["TOPIC"]["SUB_NAME"].as<std::string>();
    mQueueSize = config["TOPIC"]["QUEUE_SIZE"].as<uint32_t>();
    mXycarSpeed = config["XYCAR"]["START_SPEED"].as<PREC>();
    mXycarMaxSpeed = config["XYCAR"]["MAX_SPEED"].as<PREC>();
    mXycarMinSpeed = config["XYCAR"]["MIN_SPEED"].as<PREC>();
    mXycarSpeedControlThreshold = config["XYCAR"]["SPEED_CONTROL_THRESHOLD"].as<PREC>();
    mAccelerationStep = config["XYCAR"]["ACCELERATION_STEP"].as<PREC>();
    mDecelerationStep = config["XYCAR"]["DECELERATION_STEP"].as<PREC>();
    mDebugging = config["DEBUG"].as<bool>();

    double cm[]={config["CALIBRATE"]["fx"].as<double>(), 0., config["CALIBRATE"]["cx"].as<double>(), 0., config["CALIBRATE"]["fy"].as<double>(), config["CALIBRATE"]["cy"].as<double>(), 0., 0., 1. };
    double dm[]={config["CALIBRATE"]["k1"].as<double>(), config["CALIBRATE"]["k2"].as<double>(), config["CALIBRATE"]["p1"].as<double>(), config["CALIBRATE"]["p2"].as<double>(), config["CALIBRATE"]["k3"].as<double>()};

    cameraMatrix = cv::Mat(3, 3, cv::CV_64FC1, (void*)cm);
    distCoeffs = cv::Mat(1, 5, cv::CV_64FC1, (void*)dm);

}

template <typename PREC>
LaneKeepingSystem<PREC>::~LaneKeepingSystem()
{
    delete mPID;
    delete mPID_CURVE;
    delete mMovingAverage;
    // delete your LaneDetector if you add your LaneDetector.
}

template <typename PREC>
void LaneKeepingSystem<PREC>::run()
{
    ros::Rate rate(kFrameRate);
    while (ros::ok())
    {
        ros::spinOnce();
        /*
        write your code.
        */

        std::pair<double, std::pair<double, double>> result;
        int32_t pos_diff, filtering_result, pid_result;
        const int32_t angle_low_threshold = 50;
        const int32_t angle_high_threshold = 150;

        result = mLaneDetector->Hough(mFrame);
        pos_diff = result.first;
        // prev_result = result.second;

        // std::cout << "result : " << pos_diff << ", " << prev_result.first << ", " << prev_result.second << "\n";
        if (pos_diff == -320)
        {
            continue;
        }
        mMovingAverage->addSample(pos_diff);
        filtering_result = mMovingAverage->getResult();

        if (abs(filtering_result) >= angle_high_threshold)
        {
            pid_result = mPID_CURVE->getControlOutput(filtering_result * 2.4);
        }
        else if (abs(filtering_result) >= angle_low_threshold)
        {
            pid_result = mPID->getControlOutput(filtering_result * 1.7);
        }
        else
        {
            pid_result = mPID->getControlOutput(filtering_result);
        }
        // std::cout << "pid_result : " << pid_result << "\n";

        if (pid_result == 0)
        {
            continue;
        }
        else
        {
            speedControl(pid_result);
            drive(pid_result);
        }
    }
}

template <typename PREC>
void LaneKeepingSystem<PREC>::imageCallback(const sensor_msgs::Image& message)
{
    cv::Mat src = cv::Mat(message.height, message.width, CV_8UC3, const_cast<uint8_t*>(&message.data[0]), message.step);
    cv::cvtColor(src, mFrame, cv::COLOR_RGB2BGR);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::speedControl(PREC steeringAngle)
{
    if (std::abs(steeringAngle) > mXycarSpeedControlThreshold)
    {
        mXycarSpeed -= mDecelerationStep;
        mXycarSpeed = std::max(mXycarSpeed, mXycarMinSpeed);
        return;
    }

    mXycarSpeed += mAccelerationStep;
    mXycarSpeed = std::min(mXycarSpeed, mXycarMaxSpeed);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::drive(PREC steeringAngle)
{
    xycar_msgs::xycar_motor motorMessage;
    motorMessage.angle = std::round(steeringAngle);
    motorMessage.speed = std::round(mXycarSpeed);
    // std::cout << "motor message: " << motorMessage << "\n";

    mPublisher.publish(motorMessage);
}

template class LaneKeepingSystem<float>;
template class LaneKeepingSystem<double>;
} // namespace Xycar
