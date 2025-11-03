#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <object_image_path>" << std::endl;
        return -1;
    }

    cv::Mat objectImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (objectImg.empty()) {
        std::cout << "Could not load object image: " << argv[1] << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Could not open camera" << std::endl;
        return -1;
    }

    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();

    std::vector<cv::KeyPoint> objectKeypoints;
    cv::Mat objectDescriptors;
    sift->detectAndCompute(objectImg, cv::noArray(), objectKeypoints, objectDescriptors);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    cv::namedWindow("Object Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Object Detection", 1280, 720);

    const int MIN_GOOD_MATCHES = 10;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Failed to capture frame" << std::endl;
            break;
        }

        cv::Mat frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> frameKeypoints;
        cv::Mat frameDescriptors;
        sift->detectAndCompute(frameGray, cv::noArray(), frameKeypoints, frameDescriptors);

        std::vector<std::vector<cv::DMatch>> knnMatches;
        if (!frameDescriptors.empty()) {
            matcher->knnMatch(objectDescriptors, frameDescriptors, knnMatches, 2);
        }

        std::vector<cv::DMatch> goodMatches;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i].size() >= 2) {
                const float ratio = 0.75f;
                if (knnMatches[i][0].distance < ratio * knnMatches[i][1].distance) {
                    goodMatches.push_back(knnMatches[i][0]);
                }
            }
        }

        if (goodMatches.size() >= MIN_GOOD_MATCHES) {
            std::vector<cv::Point2f> objectPoints;
            std::vector<cv::Point2f> framePoints;

            for (size_t i = 0; i < goodMatches.size(); i++) {
                objectPoints.push_back(objectKeypoints[goodMatches[i].queryIdx].pt);
                framePoints.push_back(frameKeypoints[goodMatches[i].trainIdx].pt);
            }

            cv::Mat H = cv::findHomography(objectPoints, framePoints, cv::RANSAC);

            if (!H.empty()) {
                std::vector<cv::Point2f> objectCorners(4);
                objectCorners[0] = cv::Point2f(0, 0);
                objectCorners[1] = cv::Point2f(objectImg.cols, 0);
                objectCorners[2] = cv::Point2f(objectImg.cols, objectImg.rows);
                objectCorners[3] = cv::Point2f(0, objectImg.rows);

                std::vector<cv::Point2f> sceneCorners(4);
                cv::perspectiveTransform(objectCorners, sceneCorners, H);

                cv::line(frame, sceneCorners[0], sceneCorners[1], cv::Scalar(0, 255, 0), 4);
                cv::line(frame, sceneCorners[1], sceneCorners[2], cv::Scalar(0, 255, 0), 4);
                cv::line(frame, sceneCorners[2], sceneCorners[3], cv::Scalar(0, 255, 0), 4);
                cv::line(frame, sceneCorners[3], sceneCorners[0], cv::Scalar(0, 255, 0), 4);
                
                cv::putText(frame, 
                            "Matches: " + std::to_string(goodMatches.size()), 
                            cv::Point(10, 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 
                            1.0, 
                            cv::Scalar(0, 255, 0), 
                            2);
            }
        }

        cv::drawKeypoints(frame, frameKeypoints, frame, cv::Scalar(0, 0, 255));

        cv::Mat matchesImg;
        cv::drawMatches(objectImg, objectKeypoints, frame, frameKeypoints, goodMatches, matchesImg, 
                        cv::Scalar::all(-1), cv::Scalar::all(-1), 
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::imshow("Object Detection", matchesImg);

        if (cv::waitKey(10) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}