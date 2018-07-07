#include "Common.h"

using namespace std;

std::map<int, std::string> ReadLabelsFromPbtxt(std::string labelFilePath,
                                                              string item_id_name,
                                                              string item_display_name)
{
    std::ifstream infile(labelFilePath.c_str());
    if(!infile.is_open())
        LOG(FATAL) << "Failed to read label.pbtxt file!";

    string buff;
    std::map<int, std::string> label_map;
    while(getline(infile, buff)) {
        if(buff.find("item") != string::npos) {
            int labelId;
            string labelName;
            getline(infile, buff);
            getline(infile, buff);
            if(buff.find(item_id_name) != string::npos) {
                labelId = atoi(buff.substr(buff.find(item_id_name) + 4,
                                           buff.length() - buff.find(item_id_name) - 4).c_str());
                getline(infile, buff);
                if(buff.find(item_display_name) != string::npos) {
                    labelName = buff.substr(buff.find(item_display_name) + 15,
                                            buff.length() - buff.find(item_display_name) - 16);
                    label_map.insert(std::map<int, std::string>::value_type(labelId, labelName));
                }
            }
        }
    }

    infile.close();

    return label_map;
}

std::map<int, std::string> ReadLabelsFromTxt(std::string labelFilePath)
{
    ifstream infile(labelFilePath);
    if(!infile.is_open())
        LOG(FATAL) << "Failed to read label.txt file!";

    std::map<int, std::string> label_map;
    int labelId;
    string labelName;
    while(!infile.eof()) {
        infile >> labelId >> labelName;
        LOG(ERROR) << labelId << " " << labelName;
        if(labelName.length() > 0)
            label_map.insert(std::map<int, std::string>::value_type(labelId, labelName));
    }

    infile.close();

    return label_map;
}

std::string GetClassNameById(int class_id, std::map<int, std::string> labelMap)
{
    std::map<int, std::string>::iterator itr = labelMap.find(class_id);
    if(itr == labelMap.end())
        LOG(FATAL) << "Your label file may be wrong!";

    return itr->second;
}

void DrawBoxOnPic(const std::vector<Detection> dets, cv::Mat img)
{
    for (int i = 0; i<dets.size(); ++i) {
        cv::Rect tmp_rect = GetRectFromRect2d(
                dets[i].getRect2d(), img.cols, img.rows);
        rectangle(img, tmp_rect, cv::Scalar(255, 0, 0), 4);
        std::ostringstream os;
        os.precision(2);
        os << dets[i].getScore();
        std::string showTxt = dets[i].getClass() + "," + os.str();;
        putText(img, showTxt, // Mat, text
                cv::Point(tmp_rect.x, tmp_rect.y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
    }
}

cv::Rect GetRectFromRect2d(cv::Rect2d rect2d, int width, int height)
{
    cv::Point top_pt, bottom_pt;
    top_pt.x = int(rect2d.tl().x * width);
    top_pt.y = int(rect2d.tl().y * height);
    bottom_pt.x = int(rect2d.br().x * width);
    bottom_pt.y = int(rect2d.br().y * height);

    return cv::Rect(top_pt, bottom_pt);
}
