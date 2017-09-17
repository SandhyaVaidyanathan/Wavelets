//Daub wavelet - implementation
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\core\mat.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include<iostream>
#include<math.h>
#include<conio.h>
#include<string>

using namespace std;
using namespace cv;

Mat downSampling(Mat& img, int factor);
Mat upSampling(Mat& img, int factor);
Mat kernalCol(Mat& img, double filter[]);
Mat kernalRow(Mat& img, double filter[]);
Mat Daub(Mat &img, int k, int originalR, int originalC);
int isPowerOfTwo(unsigned int x, unsigned int y);
/// Filter values
double lpa[9] = { 0.026748757411,  -0.016864118443, -0.078223266529,
				  0.266864118443, 0.602949018236 , 0.266864118443,
				 -0.078223266529,  -0.016864118443, 0.026748757411 };

double hpa[9] = { 0, 0.091271763114, -0.057543526229, -0.591271763114,
				  1.11508705, -0.591271763114,  -0.057543526229,
				  0.091271763114, 0 };

double lps[9] = { 0,  -0.091271763114, -0.057543526229, 0.591271763114,
				  1.11508705, 0.591271763114,  -0.057543526229,
				  -0.091271763114, 0 };

double hps[9] = { 0.026748757411, -0.016864118443,  -0.078223266529 ,
				-0.266864118443, 0.602949018236, -0.266864118443,
				-0.078223266529,  -0.016864118443,  0.026748757411 };


int main()
{
	Mat inpimage;
	inpimage = imread("lena128.jpg", 0);
	Mat level;
	inpimage.copyTo(level);
	if (!inpimage.data)
	{
		return -1;
		cout << " No data entered, please enter the path to an image file" << endl;
	}

	if (!isPowerOfTwo(inpimage.rows, inpimage.cols))
	{
		return -1;
		cout << "Invalid input";
	}
	imshow("Input", inpimage);
	int choice = 0;
	int originalR = inpimage.rows;
	int originalC = inpimage.cols;
	//For the depth of iterations
	cout << "Enter the number of iterations";
	cin >> choice;
	for (int k = 0; k < choice; k++)
	{
		level = Daub(level, choice, originalR, originalC);
	}

	return 0;
}
// Applying the kernal on columns and rows
Mat kernalCol(Mat & paddedimage, double filter[])
{
	int scr = 0;
	double sum = 0;
	int colcount = 0;

	Mat tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	for (int rowcount = 0; rowcount < paddedimage.rows; rowcount++)
	{
		colcount = 0;
		while (colcount < paddedimage.cols)
		{
			if (colcount + 9 > paddedimage.cols)
				break;

			{
				for (scr = 0; scr < 9; scr++)
				{
					sum = sum + paddedimage.at<float>(rowcount, scr + colcount) * filter[scr];
				}
				tempMat.at<float>(rowcount, colcount) = sum;
				sum = 0;
				colcount++;
			}
		}
	}
	return tempMat;
}
//-------------
Mat kernalRow(Mat &paddedimage, double filter[])
{
	int rowcount = 0;
	int scr = 0;
	double sum = 0;
	Mat tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	for (int colcount = 0; colcount < paddedimage.cols; colcount++)
	{
		rowcount = 0;
		while (rowcount < paddedimage.rows)
		{
			if (rowcount + 9 > paddedimage.rows)
				break;
			for (scr = 0; scr < 9; scr++)
			{
				sum += paddedimage.at<float>(scr + rowcount, colcount) * lpa[scr];
			}
			tempMat.at<float>(rowcount, colcount) = sum;
			sum = 0;

			rowcount++;
		}
	}
	return tempMat;
}
// Major part of analysis and synthesis happens here
Mat Daub(Mat & inpimage, int choice, int originalR, int originalC)
{
	Mat paddedimage;
	Mat_<float> fm;
	inpimage.convertTo(fm, CV_32F);
	inpimage.copyTo(fm);
//The image is padded with zeros
	copyMakeBorder(fm, paddedimage, 4, 4, 4, 4, BORDER_CONSTANT, 0);
	Mat temp3;
	normalize(paddedimage, temp3, 1, 0, NORM_INF);

	// Low pass analysis + low pass analysis (down and upsampling)
	Mat tempMat = kernalCol(paddedimage, hpa);
	Mat toApplyagain;
	//copying this would be our new col conv image to be considered for row conv
	tempMat.copyTo(toApplyagain);
	//set it back to zero
	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F); 
	tempMat = kernalRow(toApplyagain, lpa);
	//Removing the black pixels on the sides
	Mat lldown = tempMat(Rect(0, 0, tempMat.cols - 8, tempMat.rows - 8));
	//downsampling by 2
	lldown = downSampling(lldown, 2);
	Mat llup = lldown(Rect(0, 0, lldown.cols, lldown.rows));
	//upsampled by 2
	llup = upSampling(llup, 2);
	//coefficient kernals are applied again
	Mat tempup = kernalCol(llup, lps);
	tempup.copyTo(llup);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalRow(llup, hps);
	Mat dst1 = tempup(Rect(0, 0, tempup.cols - 8, tempup.rows - 8));
	normalize(lldown, lldown, 1, 0, NORM_INF);
	normalize(dst1, dst1, 1, 0, NORM_INF);
	//The above process is repeated for all other combinations as well

	//highpass analysis + highpass analysis (downsampling + upsampling)
	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat = kernalCol(paddedimage, hpa);
	Mat hhTemp = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat.copyTo(hhTemp);
	// set it back to zero
	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat = kernalRow(hhTemp, hpa);
	Mat hhdown = tempMat(Rect(0, 0, tempMat.cols - 8, tempMat.rows - 8));
	hhdown = downSampling(hhdown, 2);
	Mat hhup = hhdown(Rect(0, 0, hhdown.cols, hhdown.rows));
	hhup = upSampling(hhup, 2);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalCol(hhup, hps);
	tempup.copyTo(hhup);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalRow(hhup, hps);
	Mat dst4 = tempup(Rect(0, 0, tempup.cols - 8, tempup.rows - 8));
	normalize(hhdown, hhdown, 1, 0, NORM_INF);
	normalize(dst4, dst4, 1, 0, NORM_INF);

	//high pass and lowpass

	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat = kernalCol(paddedimage, hpa);
	Mat hlTemp = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat.copyTo(hlTemp);
	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat = kernalRow(hlTemp, lpa);
	Mat hldown = tempMat(Rect(0, 0, tempMat.cols - 8, tempMat.rows - 8));
	hldown = downSampling(hldown, 2);
	Mat hlup = hldown(Rect(0, 0, hldown.cols, hldown.rows));
	hlup = upSampling(hlup, 2);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalCol(hlup, hps);
	tempup.copyTo(hlup);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalRow(hlup, lps);
	Mat dst3 = tempup(Rect(0, 0, tempup.cols - 8, tempup.rows - 8));
	normalize(hldown, hldown, 1, 0, NORM_INF);
	normalize(dst3, dst3, 1, 0, NORM_INF);

	//lowpass and highpass
	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat = kernalCol(paddedimage, lpa);
	Mat lhTemp = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat.copyTo(lhTemp);
	tempMat = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempMat = kernalRow(lhTemp, hpa);
	Mat lhdown = tempMat(Rect(0, 0, tempMat.cols - 8, tempMat.rows - 8));
	lhdown = downSampling(lhdown, 2);
	Mat lhup = lhdown(Rect(0, 0, lhdown.cols, lhdown.rows));
	lhup = upSampling(lhup, 2);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalCol(lhup, lps);
	tempup.copyTo(lhup);
	tempup = Mat::zeros(paddedimage.rows, paddedimage.cols, CV_32F);
	tempup = kernalRow(lhup, hps);
	Mat dst2 = tempup(Rect(0, 0, tempup.cols - 8, tempup.rows - 8));
	normalize(lhdown, lhdown, 1, 0, NORM_INF);
	normalize(dst2, dst2, 1, 0, NORM_INF);

// The images that are upsampled and the filters passed on are added to give the restored image
	Mat dstfinal = dst1 + dst2 + dst3 + dst4;
	imshow("final", dstfinal);

	//display deconstructed image in one frame
	Mat imd(originalR, originalC, CV_32F);
	hhdown.copyTo(imd(Rect(inpimage.rows / 2 - 1, inpimage.cols / 2 - 1, inpimage.rows / 2, inpimage.cols / 2)));
	hldown.copyTo(imd(Rect(0, inpimage.cols / 2 - 1, inpimage.rows / 2, inpimage.cols / 2)));
	lldown.copyTo(imd(Rect(inpimage.rows / 2 - 1, 0, inpimage.rows / 2, inpimage.cols / 2)));
	lhdown.copyTo(imd(Rect(0, 0, inpimage.rows / 2, inpimage.cols / 2)));

	imshow("Deconstruction", imd);

	//Save image
	dstfinal.convertTo(dstfinal, CV_32FC1, 255.0);
	imwrite("daub.jpg", dstfinal);
	cout << " The synthesized image is stored as daub.jpg";
	waitKey(0);

	return 	lhdown;
}

Mat downSampling(Mat& img, int z)
{
	Mat temp = Mat::zeros(img.rows / z, img.cols / z, CV_32F);
	for (int i = 0, x = 0; i < img.rows; i += z, x++)
	{
		for (int j = 0, y = 0; j < img.cols; j += z, y++) 
		{
			temp.at<float>(x, y) = img.at<float>(i, j);
		}
	}
	return temp;
}

Mat upSampling(Mat& img, int z) 
{
	Mat temp = Mat::zeros(img.rows*z, img.cols*z, CV_32F);
	for (int i = 0, x = 0; i < img.rows; i ++, x+=2) 
	{
		for (int j = 0, y = 0; j < img.cols; j++, y+=2) 
		{
			 if (x % 2 == 0 && y % 2 == 0)
				temp.at<float>(x, y) = img.at<float>(i, j);
		}
	}
	return temp;
}

// checks if its a valid input size
int isPowerOfTwo(unsigned int x, unsigned int y)
{
	if (x == y)
	{
		while (((x % 2) == 0) && x > 1)
			x /= 2;
		return (x == 1);
	}
	else
		cout << "Please input an image that is compatible - 2^n * 2^n";
}


