#include "Detection.h"

using namespace cv;
using namespace std;
// constructor
Detection::Detection()
{
}

Detection::Detection(const std::string &cls, const cv::Rect2d &rect2d, const float &score)
        : m_cls(cls), m_rect2d(rect2d), m_score(score)
{
}

// copy constructor
Detection::Detection(Detection const& detection)
{
    m_cls = detection.m_cls;
    m_score = detection.m_score;
    m_rect2d = detection.m_rect2d;
}

// set value constructor
Detection& Detection::operator=(const Detection &detection)
{
    m_cls = detection.m_cls;
    m_score = detection.m_score;
    m_rect2d = detection.m_rect2d;

    return *this;
}

Detection::~Detection()
{

}
//Interface
string Detection::getClass() const
{
    return m_cls;
}


float Detection::getScore() const
{
    return m_score;
}

cv::Rect2d Detection::getRect2d() const
{
    return m_rect2d;
}

void Detection::setClass(const std::string& cls)
{
    m_cls = cls;
}


void Detection::setScore(const float& score)
{
    m_score = score;
}

void Detection::setRect2d(const cv::Rect2d &rect2d)
{
    m_rect2d = rect2d;
}
