#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	// rely on $OpenCV_DIR environment variable
	std::string face_cascade_name = cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml");
	std::cout << "haar cacscade path: " + face_cascade_name << std::endl;
	// load face cascade
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "failed to load face cascade\n";
		return -1;
	}

	cv::Mat img = cv::imread("../img/the_expanse.jpg", cv::IMREAD_COLOR);

	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(img_gray, img_gray);

	// detect faces
	std::vector<cv::Rect> faces;
	// face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, 0, cv::Size(10, 10));
	face_cascade.detectMultiScale(img_gray, faces);
	std::cout << "detected " << faces.size() << " faces\n";

	for (size_t i = 0; i < faces.size(); i++) {
		std::cout << "rectangle found: " << faces[i] << std::endl;
		cv::rectangle(img, faces[i], cv::Scalar(0, 255, 0), 1);
	}

	cv::imshow("Display window", img);
	std::cout << "press any key to exit\n";
	cv::waitKey(0);

	return 0;
}

