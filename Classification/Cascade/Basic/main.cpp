#include <opencv2/opencv.hpp>
#include <vector>

int main() {
	std::string path = "/usercode/faces1.jpg";
	cv::Mat img = cv::imread(path);

	cv::Mat imgGray;
	cv::cvtColor(img,imgGray, cv::COLOR_BGR2GRAY);

	cv::CascadeClassifier faceCascade;
	faceCascade.load("/usercode/haarcascades/haarcascade_frontalface_alt2.xml");

    std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(img, faces, 1.1, 3);
	for (int i = 0; i < faces.size(); i++)
	{
		cv::rectangle(img, faces[i].tl(), faces[i].br(), cv::Scalar(255, 0, 255), 1);
	}
	cv::imshow("FaceDetection", img);
	cv::waitKey(0);
	return 0;
}