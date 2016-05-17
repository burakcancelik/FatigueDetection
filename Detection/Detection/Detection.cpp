// Detection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);

String face_cascade_name = "C:\\opencv11\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
String eyes_cascade_name = "C:\\opencv11\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml";
String mouth_cascade_name = "C:\\opencv11\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_mouth.xml";
CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;
CascadeClassifier mouth_cascade;
string window_name = "Face-Eye-Mouth Detection";


int main(int argc, const char ** argv)
{
	CvCapture* capture;
	Mat frame;

	//CvCapture works fine in C,VideoCapture works fine with C++

	if (!face_cascade.load(face_cascade_name)){ cout << "Face cascade cannot load\n"; };
	if (!eye_cascade.load(eyes_cascade_name)){ cout << "Eye cascade cannot load\n"; };
	if (!mouth_cascade.load(mouth_cascade_name)){ cout << "Mouth cascade cannot load\n"; };

	capture = cvCaptureFromCAM(0);

	//double fps = capture.get(CV_CAP_PROP_FPS);

	if (capture)
	{
		while (true)
		{
			frame = cvQueryFrame(capture);
			//cout << frame.data;

			if (!frame.empty())
			{
				detectAndDisplay(frame);
			}
			else{ printf("No captured frame"); break; };

			int wait = waitKey(10);
			if (char(wait) == 'q'){ break; }
		}
	}
	return 0;
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	std::vector<Rect> faces_for_mouth;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	/*face_cascade.detectMultiScale(frame_gray, faces_for_mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));*/

	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		//----------------------------------------Eye Detection -----------------------------------------------------
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

		for (int j = 0; j < eyes.size(); j++)
		{
			if (!eyes.empty())
			{
				Point center(faces[i].x + eyes[j].x + eyes[i].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.1);
				//void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
				circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			}

		}



		//--------------------------------------------- Mouth Detection --------------------------------------
		try
		{
			std::vector<Rect>mouth;
			//minimum detection -> Size()
			mouth_cascade.detectMultiScale(faceROI, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

			if (!mouth.empty())
			{
				Point center2(faces[0].x + mouth[0].x + mouth[0].width / 2, faces[0].y + mouth[0].y + mouth[0].height / 2);
				int yawn = cvRound((mouth[0].width + mouth[0].height)*0.15);
				circle(frame, center2, yawn, Scalar(0, 255, 0), 3, 8, 0);
			}

		}

		catch (exception e)
		{
			continue;
		}

	}
	imshow(window_name, frame);
}