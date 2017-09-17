//haar wavelet - implementation
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\core\mat.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include<iostream>
#include<math.h>
#include<conio.h>
#include "opencv2/imgcodecs.hpp"
using namespace std;
using namespace cv;
int isPowerOfTwo(unsigned int x, unsigned int y);
int main()
{
	Mat HT, GT, P11, P12, P13, P14, temp, dummy;
	Mat grayImage;

	Mat imgOriginal = cv::imread("lena128.jpg");          // open image
	if (imgOriginal.empty()) {                                  // if unable to open image
		cout << "error: image not read from file\n\n";     // show error message on command line
		_getch();                                            
		return(0);                                             
	}
	if (!isPowerOfTwo(imgOriginal.rows, imgOriginal.cols))
	{
		return -1;
		cout << "Invalid input";
	}
	cvtColor(imgOriginal, grayImage, CV_BGR2GRAY);       // convert to grayscale
	namedWindow("imgOriginal", WINDOW_AUTOSIZE);
	imshow("imgOriginal", imgOriginal);
	int rc = imgOriginal.rows / 2;
	int cc = imgOriginal.cols;
	const int r = 64;
	const int c = 128;
	int size = 0;
	bool x[c] = { false };
	int count = 0;
	float G[r][c];
	float H[r][c];
// H and G filter values
	for (int i = 0; i < r; i++)
	{
		size = count + 2;
		for (int j = 0; j < c; j++)
		{
			if (j < size && !x[j])
			{
				G[i][j] = 0.707;
				x[j] = true;
				count++;
			}
			else
			{
				G[i][j] = 0.0;
			}
		}
	}
	cv::Mat Gfilter = cv::Mat(r, c, CV_32FC1, G);
	size = 0;
	bool negative[c] = { false };
	count = 0;
	for (int i = 0; i < r; i++)
	{
		size = count + 2;
		for (int j = 0; j < c; j++)
		{
			if (j < size && !negative[j])
			{
				if (j % 2 == 0)
					H[i][j] = 0.707;
				else
					H[i][j] = -0.707;
				negative[j] = true;
				count++;
			}
			else
				H[i][j] = 0.0;
		}
	}
	cv::Mat Hfilter = cv::Mat(r, c, CV_32FC1, H);
	//Transpose HT and GT
	cv::transpose(Hfilter, HT);
	cv::transpose(Gfilter, GT);

	// P11, P12, P13, P14
	grayImage.convertTo(grayImage, CV_32FC1);
	cv::gemm(Hfilter, grayImage, 1.0, dummy, 0.0, temp);
	cv::gemm(temp, HT, 1.0, dummy, 0.0, P11);
	//P12
	cv::gemm(Hfilter, grayImage, 1.0, dummy, 0.0, temp);
	cv::gemm(temp, GT, 1.0, dummy, 0.0, P12);
	//P13
	cv::gemm(Gfilter, grayImage, 1.0, dummy, 0.0, temp);
	cv::gemm(temp, HT, 1.0, dummy, 0.0, P13);
	//P14
	cv::gemm(Gfilter, grayImage, 1.0, dummy, 0.0, temp);
	cv::gemm(temp, GT, 1.0, dummy, 0.0, P14);
	// combine in one window
	Mat imd = Mat::zeros(imgOriginal.rows, imgOriginal.cols, CV_32F);
	P11.copyTo(imd(Rect(imgOriginal.rows/2 -1, imgOriginal.cols/2 -1, imgOriginal.rows/2, imgOriginal.cols/2)));
	P12.copyTo(imd(Rect(0, imgOriginal.cols/2 -1, imgOriginal.rows/2, imgOriginal.cols/2)));
	P13.copyTo(imd(Rect(imgOriginal.rows/2 -1, 0, imgOriginal.rows/2, imgOriginal.cols/2)));
	P14.copyTo(imd(Rect(0, 0, imgOriginal.rows/2, imgOriginal.cols/2)));
	// Normalised for better view, as the numbers are floating point
	Mat tem2;
	normalize(imd, tem2, 1, 0, NORM_INF);
	namedWindow("Decomposition", WINDOW_AUTOSIZE);
	imshow("Decomposition", tem2);
	//binary file write
	// snippet from the web
	std::vector<float> array;
	if (tem2.isContinuous()) {
		array.assign((float*)tem2.datastart, (float*)tem2.dataend);
	}
	else {
		for (int i = 0; i < tem2.rows; ++i) {
			array.insert(array.end(), (float*)tem2.ptr<uchar>(i), (float*)tem2.ptr<uchar>(i) + tem2.cols);
		}
	}
	cv::FileStorage fs("analysis1.wl", cv::FileStorage::WRITE);
	fs << "Haar" << array;
	//binary file fwrite
	FILE *f = fopen("analysis2.wl", "wb");
	while (1)
	{
			fwrite(tem2.data, sizeof(float), tem2.rows * tem2.cols, f);
			break;
	}
	fclose(f);
	cout << " The files are stored as analysis1.wl and analysis2.wl";
	//Synthesis--------------------------------------------------
	Mat H1, H2n, H2d, H2, H3, P_reconstructed;
	hconcat(HT, GT, H1);
	vconcat(Hfilter, Gfilter, H3);

	hconcat(P11, P12, H2n);
	hconcat(P13, P14, H2d);
	vconcat(H2n, H2d, H2);

	cv::gemm(H1, H2, 1.0, dummy, 0.0, temp);
	cv::gemm(temp, H3, 1.0, dummy, 0.0, P_reconstructed);
	//normalize
	Mat temp3;
	normalize(P_reconstructed, temp3, 1, 0, NORM_INF);
	namedWindow("Reconstructed", WINDOW_AUTOSIZE);
	imshow("Reconstructed", temp3);

	//Save image
	imwrite("haar.jpg", P_reconstructed);
	cout << " The synthesized image is stored as haar.jpg" <<endl;

	cv::waitKey(0);

	return 0;
}

// checks if the image is 2^n*2^n
int isPowerOfTwo(unsigned int x, unsigned int y)
{
	if (x == y)
	{
		while (((x % 2) == 0) && x > 1)
			x /= 2;
		cout << "Your input image is valid" << endl;
		return (x == 1);

	}
	else
		cout << "Please input an image that is compatible - 2^n * 2^n"<<endl;
}

