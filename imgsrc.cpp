#include"imghead.h"

void deNoiseTem(vector<Mat> &temImgVec, Mat src, Mat &dst, Mat &dif, 
	                     Mat &difR, Mat &difG, Mat &difB)
{
	temImgVec.push_back(src);
	dst = src.clone();

	if (temImgVec.size() < 2)
	{
		return ;
	}
	else
	{   
		Mat imgSrc = src.clone();
		Mat srcGray;
		cvtColor(imgSrc, srcGray, CV_BGR2GRAY);
		//blur(srcGray, srcGray, Size(3, 3));
		//Mat edges = srcGray.clone();
		//Sobel(srcGray, edges, srcGray.depth(), 1, 1, 3);
		//namedWindow("Edge", 0);
		//imshow("Edge", edges);

		Mat lastFrame = temImgVec[0].clone();
		Mat lastGray;
		cvtColor(lastFrame, lastGray, CV_BGR2GRAY);

		//blur(lastGray, lastGray, Size(3, 3));
		//Mat LastEdges = lastGray.clone();
		//Sobel(lastGray, LastEdges, lastGray.depth(), 1, 1, 3);
		//namedWindow("LastEdges", 0);
		//imshow("LastEdges", LastEdges);

		//Mat difEdges(LastEdges.size(), CV_8UC1, Scalar(0));

		//for (int i = 0; i < LastEdges.rows; i++)
		//{
		//	for (int j = 0; j < LastEdges.cols; j++)
		//	{
		//		int difV = edges.at<uchar>(i, j) - LastEdges.at<uchar>(i, j);
		//		difV = difV < 0 ? 0 : difV;
		//		difEdges.at<uchar>(i, j) = difV;
		//	}
		//}
		//medianBlur(difEdges, difEdges, 3);
		//threshold(difEdges, difEdges, 5, 255, 0);
		//medianBlur(difEdges, difEdges, 3);
		//namedWindow("difEdges", 0);
		//imshow("difEdges", difEdges);




		int dstHeight = dst.rows;
		int dstWith = dst.cols;
		float weight[2] = { 0 };

		for (int h = 0; h < dstHeight; h++)
		{
			for (int w = 0; w < dstWith; w++)
			{
				int Sad = 0;
				int sadr = temImgVec[0].at<Vec3b>(h, w)[0] - src.at<Vec3b>(h, w)[0];
				int sadg = temImgVec[0].at<Vec3b>(h, w)[1] - src.at<Vec3b>(h, w)[1];
				int sadb = temImgVec[0].at<Vec3b>(h, w)[2] - src.at<Vec3b>(h, w)[2];

				difR.at<uchar>(h, w) = abs(sadr);
				difG.at<uchar>(h, w) = abs(sadg);
				difB.at<uchar>(h, w) = abs(sadb);

				Sad = abs(sadr) + abs(sadg) + abs(sadb);
				int sadLimt = Sad;
				dif.at<uchar>(h, w) = RANGE_LIMIT(sadLimt);

				//if (Sad < 30)
				//{
				//	dst.at<Vec3b>(h, w)[0] = temImgVec[0].at<Vec3b>(h, w)[0];
				//	dst.at<Vec3b>(h, w)[1] = temImgVec[0].at<Vec3b>(h, w)[1];
				//	dst.at<Vec3b>(h, w)[2] = temImgVec[0].at<Vec3b>(h, w)[2];
				//}

				weight[0] = 0.5f * (factor * Sad);
				weight[0] = weight[0] > 0.5f ? 0.5f : weight[0];
				weight[0] = 0.5f - weight[0];
				weight[1] = 1 - weight[0];

				for (int i = 0; i < 3; i++)
				{
					dst.at<Vec3b>(h, w)[i] = 0;

					dst.at<Vec3b>(h, w)[i] = static_cast<unsigned char>(weight[0] * temImgVec[0].at<Vec3b>(h, w)[i]
						                    + weight[1] * src.at<Vec3b>(h, w)[i]);
				}
			}
		}

		//dst = src.clone();

		temImgVec.pop_back();
		temImgVec.push_back(dst);// update the last element to the filtered;

		vector<Mat>::iterator k = temImgVec.begin();
		temImgVec.erase(k);//删除第一个元素
	}
	return;
}


void spannarDeNoise(Mat srcMat, Mat &dst, int radius, int maxCov)
{
	Mat srcRef;
	Mat channel[3];
	Mat chaneelBlur[3];
	split(srcMat, channel);
	srcMat.convertTo(srcRef, CV_64FC1);
	Mat mean_src, mean_src2, mean_Ip, mean_II;
	boxFilter(srcRef, mean_src, CV_64FC1, Size(radius, radius));
	boxFilter(srcRef.mul(srcRef), mean_src2, CV_64FC1, Size(radius, radius));
	Mat var_I = mean_src2 - mean_src.mul(mean_src);

	boxFilter(channel[0], chaneelBlur[0], CV_8UC1, Size(radius, radius));
	boxFilter(channel[1], chaneelBlur[1], CV_8UC1, Size(radius, radius));
	boxFilter(channel[2], chaneelBlur[2], CV_8UC1, Size(radius, radius));

	Mat f1 = srcRef.clone();
	f1 = var_I / maxCov;

	for (int i = 0; i < srcMat.rows; i++)
	{
		for (int j = 0; j < srcMat.cols; j++)
		{
			if (f1.at<double>(i, j) > 1)
			{
				f1.at<double>(i, j) = 1;
			}
			double c1 = f1.at<double>(i, j);
			double c2 = 1 - c1;

			dst.at<Vec3b>(i, j)[0] = c1 * srcMat.at<Vec3b>(i, j)[0] + c2 * chaneelBlur[0].at<uchar>(i, j);
			dst.at<Vec3b>(i, j)[1] = c1 * srcMat.at<Vec3b>(i, j)[1] + c2 * chaneelBlur[1].at<uchar>(i, j);
			dst.at<Vec3b>(i, j)[2] = c1 * srcMat.at<Vec3b>(i, j)[2] + c2 * chaneelBlur[2].at<uchar>(i, j);
		}
	}
}

void contrastEnhance(Mat src, Mat &dst)//bgr
{   
	dst = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{   
			//// b > (g + r)/2
			int color_b = src.at<Vec3b>(i, j)[0];
			int color_g = src.at<Vec3b>(i, j)[1];
			int color_r = src.at<Vec3b>(i, j)[2];
			//int averColor_gr = (color_g + color_r) / 2;
			//if (color_b > averColor_gr)
			//{
			//	dst.at<Vec3b>(i, j)[0] = averColor_gr;
			//}

			int tip = 10;
			int dstColor_b = color_b - tip;
			int dstColor_g = color_g - tip;
			int dstColor_r = color_r + 2 * tip;
			RANGE_LIMIT(dstColor_b);
			RANGE_LIMIT(dstColor_g);
			RANGE_LIMIT(dstColor_r);
			dst.at<Vec3b>(i, j)[0] = dstColor_b;
			dst.at<Vec3b>(i, j)[1] = dstColor_g;
			dst.at<Vec3b>(i, j)[2] = dstColor_r;
		}
	}
}