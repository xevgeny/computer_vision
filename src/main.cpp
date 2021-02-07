#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace cv;
using namespace face;

const Size target_size = Size(250, 250);

int main()
{
    // rely on $OpenCV_DIR environment variable
    std::string face_cascade_name = samples::findFile("haarcascades/haarcascade_frontalface_default.xml");
    std::cout << "haar cacscade path: " + face_cascade_name << std::endl;
    // load face cascade
    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_name))
    {
        std::cout << "failed to load face cascade\n";
        return -1;
    }

    // load eigenfaces model
    Ptr<EigenFaceRecognizer> face_recognizer = EigenFaceRecognizer::create();
    face_recognizer->read("../eigenfaces.model");

    Mat img = imread("../img/the_expanse.jpg", IMREAD_COLOR);

    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);

    // detect faces
    std::vector<Rect> faces;
    // face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, 0, Size(10, 10));
    face_cascade.detectMultiScale(img_gray, faces);
    std::cout << "detected " << faces.size() << " faces\n";

    for (size_t i = 0; i < faces.size(); i++)
    {
        std::cout << "rectangle found: " << faces[i] << std::endl;
        rectangle(img, faces[i], Scalar(0, 0, 255), 1);

        // face recognition using eigenface model
        Mat img_roi = Mat(img_gray, faces[i]);
        Mat img_roi_resized; 
        resize(img_roi, img_roi_resized, target_size);
        std::cout << img_roi_resized.size << std::endl;
        
        int predicted_label = -1;
        double predicted_confidence = 0.0;
        face_recognizer->predict(img_roi_resized, predicted_label, predicted_confidence);
        std::cout << predicted_label << ", " << predicted_confidence << std::endl;
    }

    imshow("Display window", img);
    std::cout << "press any key to exit\n";
    waitKey(0);

    return 0;
}
