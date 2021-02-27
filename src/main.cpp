#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

void detectFaces(CascadeClassifier &cc, const Mat &frame)
{
    Mat grayFrame;
    std::vector<Rect> faces;

    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    // 10% and 90% of the frame height
    int minSize = frame.size[0] * 0.1;
    int maxSize = frame.size[0] * 0.9;

    cc.detectMultiScale(grayFrame,
                        faces,
                        1.1,
                        3,
                        0,
                        Size(minSize, minSize),
                        Size(maxSize, maxSize));

    for (size_t i = 0; i < faces.size(); i++)
    {
        std::stringstream ss;
        ss << "Face: " << i << " " << faces[i];
        std::cout << ss.str() << std::endl;
        putText(frame,
                ss.str(),
                Point(50, 100 + (i * 50)),
                FONT_HERSHEY_SIMPLEX,
                1,
                CV_RGB(255, 255, 255),
                1);
        rectangle(frame, faces[i], CV_RGB(255, 0, 0), 1);
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "unexpected number of arguments" << std::endl;
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cout << "error opening video capture: " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "FPS: " << cap.get(CAP_PROP_FPS) << std::endl;
    std::cout << "frames: " << cap.get(CAP_PROP_FRAME_COUNT) << std::endl;

    // rely on $OpenCV_DIR environment variable
    CascadeClassifier faceCascade;
    std::string faceCascadePath = samples::findFile("haarcascades/haarcascade_frontalface_alt2.xml");
    if (!faceCascade.load(faceCascadePath))
    {
        std::cout << "failed to load face cascade\n";
        return -1;
    }

    int currentFrame = 0;
    cap.set(CAP_PROP_POS_FRAMES, currentFrame);

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (!cap.read(frame))
            break;

        ++currentFrame;
        // process every 5th frame
        if (currentFrame % 5 != 0)
            continue;

        std::cout << "Frame: " << currentFrame << std::endl;

        detectFaces(faceCascade, frame);

        putText(frame,
                "Frame: " + std::to_string(currentFrame),
                Point(50, 50),
                FONT_HERSHEY_SIMPLEX,
                1,
                CV_RGB(255, 255, 255),
                1);

        imshow("Video", frame);

        // press ESC to exit
        char c = (char)waitKey(5);
        if (c == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}