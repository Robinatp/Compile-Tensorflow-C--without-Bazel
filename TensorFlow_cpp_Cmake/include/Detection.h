#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <opencv2/core/core.hpp>
#include "tensorflow/core/platform/logging.h"

class Detection
{
public:
    Detection();
//    Detection(const std::string& cls, const cv::Rect& rect, const float& score);
    Detection(const std::string& cls, const cv::Rect2d& rect2d, const float& score);
    Detection(Detection const& detection);
    Detection& operator=(const Detection &detection);

    ~Detection();

    //Interface
    std::string getClass() const;
    float getScore() const;
    cv::Rect2d getRect2d() const;

    void setClass(const std::string& cls);
    void setScore(const float& score);
    void setRect2d(const cv::Rect2d& rect2d);

private:
    std::string m_cls;
    float m_score;
    cv::Rect2d m_rect2d;

};

#endif // DETECTION_H
