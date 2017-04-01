//#include <opencv2\core.hpp>
//#include <opencv2\highgui.hpp>
//
////#include <iostream>
//
//using namespace std;
//
//void knnSegmentation(cv::Mat input_img, cv::Mat clusters);
//
//void calcClusterPlanesLp(cv::Mat clusters, unsigned p);
//
//void print_help()
//{
//	printf("Program usage: ...\n");
//}
//
//int main(int argc, char **argv)
//{
//	if (argc > 1)
//	{
//		cv::Mat input_img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
//		cv::Size img_size = input_img.size();
//		int width = img_size.width, height = img_size.height;
//		int img_type = input_img.type();
//
//		printf("Image type: %i\n", img_type);
//
//		cv::Mat img_float = cv::Mat(img_size, CV_32F);
//		if (img_type != CV_32F)
//		{
//			input_img.convertTo(img_float, CV_32F, 1.f / 255);
//		}
//		else
//		{
//			input_img.copyTo(img_float);
//		}
//
//		cv::Mat clusters;
//		if (atoi(argv[2]) == 0)
//			knnSegmentation(img_float, clusters);
//		else
//		{
//			printf("Invalid segmentation parameter\n");
//			system("pause");
//			exit(1);
//		}
//
//		if (atoi(argv[3]) == 0)
//			calcClusterPlanesLp(clusters, 2);
//		else if (atoi(argv[3]) == 1)
//			calcClusterPlanesLp(clusters, 1);
//		else
//		{
//			printf("Invalid plane calculation parameter\n");
//			system("pause");
//			exit(2);
//		}
//
//	}
//	else
//		print_help();
//
//	system("pause");
//	return 0;
//}
//
//void knnSegmentation(cv::Mat input_img, cv::Mat clusters)
//{
//	//int num_clusters = 10;
//	//cv::kmeans(input_img, k, 
//}
//
//void calcClusterPlanesLp(cv::Mat clusters, unsigned p)
//{
//
//}