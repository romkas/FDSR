#include "disjointSetClass.h"
#include "Kruskal.h"



#include <iostream>
//#include <fstream>
#include <cstring>

using namespace std;

void print_help()
{
	printf("Program usage:\narg1 - size of pixel vicinity (4, 8, 24, 48)\narg2 - metric function for edge weight calculation\n"
		"arg3 - Kruskal k parameter\narg4 - segment size threshold\narg5 - target number of segments\n"
		"arg6 - z");
#if USE_COLOR == 1
	printf("color image file path");
#else
	printf("grayscale image file path");
#endif
	printf("\narg7 - is depth map data given\narg8 - z coordinate scaling\narg9 - depth data file path");
}

void ReadPFMFile(cv::Mat& img, const char* filename)
{
	char buf[12], tmp;
	FILE *f = fopen(filename, "rb");
	fscanf(f, "%s\n", buf);
	int img_type = (strcmp(buf, "Pf") == 0) ? CV_32FC1 : CV_32FC3;
	int nc = (img_type == CV_32FC1) ? 1 : 3;
	int w, h;
	fscanf(f, "%d %d\n", &w, &h);
	float scale;
	fscanf(f, "%f\n", &scale);
	int little_endian = 0;
	if (scale < 0.0f) {
		little_endian = 1;
		scale = -scale;
	}
	img = cv::Mat(cv::Size(w, h), img_type);
	for (int i = h - 1; i >= 0; i--)
		for (int j = 0; j < w; j++)
		{
			if (fread(buf, sizeof(char), 4 * nc, f) != 4 * nc)
			{
				printf("Error reading PFM file.\n");
				exit(3);
			}
			if (little_endian)
				for (int c = 0; c < nc; c++)
				{
					tmp = buf[c * 4 + 3];
					buf[c * 4 + 3] = buf[c * 4];
					buf[c * 4 + 3] = tmp;
					tmp = buf[c * 4 + 2];
					buf[c * 4 + 2] = buf[c * 4 + 1];
					buf[c * 4 + 2] = tmp;
				}
			if (nc == 1)
				img.at<float>(i, j) = *((float *)buf);
			else
				img.at<cv::Vec3f>(i, j) = cv::Vec3f(*((float *)buf), *((float *)(buf + 4)), *((float *)(buf + 8)));
		}
	fclose(f);
}

//void plot_image(cv::Mat &img, int im_type, char *title)
//{
//
//}

//int resize_image(cv::Mat &image_in, cv::Mat &out, size_t w, size_t h)
//{
//	int w = img.size().width, h = img.size().height;
//	if (!strcmp(argv[3], "2"))
//	{
//		w /= 2; h /= 2;
//	}
//	else if (!strcmp(argv[3], "4"))
//	{
//		w /= 4; h /= 4;
//	}
//	else if (!strcmp(argv[3], "0"))
//	{
//		w = atoi(argv[4]); h = atoi(argv[5]);
//	}
//	else
//		exit(3);
//	cv::Size imsz = cv::Size(w, h);
//	cv::resize(img, img_out, imsz, 0, 0, CV_INTER_LINEAR);
//}

int main(int argc, char **argv)
{
	if (argc >= 8)
	{
		int c = 1;
		int param_pixel_vicinity = std::atoi(argv[c++]);
		int param_metrics_flag = std::atoi(argv[c++]);
		double param_k = std::atof(argv[c++]);
		int param_min_segment_size = std::atoi(argv[c++]);
		int param_target_num_segments = std::atoi(argv[c++]);
		//int param_segment_size_vis = std::atoi(argv[c++]);
		//bool param_color = (bool)std::atoi(argv[c++]);
		double param_z_coord_weight;

		cv::Mat img, img_float, depth;
#if USE_COLOR == 1
		img = cv::imread(argv[c++], cv::IMREAD_COLOR);
#else
		img = cv::imread(argv[c++], cv::IMREAD_GRAYSCALE);
#endif

		/*cv::namedWindow("source image-1", cv::WINDOW_AUTOSIZE);
		cv::imshow("source image-1", img);
		cv::waitKey();*/

		int target_w = 320, target_h = 240;
		cv::resize(img, img, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
		cv::Size img_size = img.size();
		int width = img_size.width, height = img_size.height;
		int img_type = img.type();

		double _min, _max;

		bool param_depthdata = (bool)std::atoi(argv[c++]);
		if (param_depthdata)
		{
			ReadPFMFile(depth, argv[c++]);
			param_z_coord_weight = std::atof(argv[c++]);
		}
		else
		{
			depth = cv::Mat::zeros(img_size, img_type);
			param_z_coord_weight = 1.;
		}

		if (param_depthdata)
			cv::resize(depth, depth, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

		if (img.depth() != CV_32F)
#if USE_COLOR == 1
			img.convertTo(img_float, CV_32FC3);
#else
			img.convertTo(img_float, CV_32FC1);
#endif
		else
			img_float = img;

		ImageGraph G0 = ImageGraph(img_float, depth, param_pixel_vicinity, param_metrics_flag, param_z_coord_weight);
		int n_segments;
		n_segments = G0.SegmentationKruskal(param_k);

		printf("Found segments: %7i\n", n_segments);

		int pixels_under_thres, seg_under_thres, num_mergers;
		G0.Clustering(param_min_segment_size, param_target_num_segments, &pixels_under_thres, &seg_under_thres, &num_mergers);

		{
			cv::Mat img_to_plot = cv::Mat(img);
			cv::namedWindow("source image", cv::WINDOW_AUTOSIZE);
			cv::imshow("source image", img_to_plot);
			cv::waitKey(100);
			if (param_depthdata)
			{
				cv::minMaxIdx(depth, &_min, &_max);
				img_to_plot = (depth - (float)_min) / ((float)_max - (float)_min);
				cv::namedWindow("source depth", cv::WINDOW_AUTOSIZE);
				cv::imshow("source depth", img_to_plot);
				cv::waitKey(100);
			}
		}

		G0.PlotSegmentation(100, "segmented");

		cv::Mat img_blurred(img_float.size(), img_float.type()), depth_blurred;

		clock_t t = clock();
		cv::GaussianBlur(img_float, img_blurred, cv::Size(5, 5), 0.7, 0.7, cv::BORDER_CONSTANT);
		t = clock() - t;
		printf("TIME (Gaussian blur. Src                  ) (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

		if (param_depthdata)
			depth_blurred = cv::Mat(depth.size(), depth.type());
		
		
		cv::Mat depth_blurred2(depth.size(), depth.type());
		t = clock();
		cv::GaussianBlur(depth, depth_blurred, cv::Size(3, 3), 0.7, 0.7, cv::BORDER_CONSTANT);
		cv::GaussianBlur(depth, depth_blurred2, cv::Size(7, 7), 0.7, 0.7, cv::BORDER_CONSTANT);
		depth += cv::abs(depth_blurred2 - depth_blurred);
		cv::minMaxIdx(depth, &_min, &_max);
		for (int i = 0; i < depth.rows; i++)
			for (int j = 0; j < depth.cols; j++)
			{
				if (depth.at<float>(i, j) > _max)
					depth.at<float>(i, j) = (float)_max;
			}
		t = clock() - t;
		printf("TIME (Gaussian blur and DoG. Depth          (ms): %8.2f\n", (double)t * 1000. / CLOCKS_PER_SEC);

		{
			cv::Mat img_to_plot;
			cv::minMaxIdx(depth, &_min, &_max);
			img_to_plot = (depth - (float)_min) / ((float)_max - (float)_min);
			cv::namedWindow("source depth", cv::WINDOW_AUTOSIZE);
			cv::imshow("source depth", img_to_plot);
			cv::waitKey(100);
		}
		
		
		
		{
			cv::Mat img_to_plot = cv::Mat(img_blurred);
			cv::Mat p[3];
			cv::split(img_to_plot, p);
			cv::minMaxIdx(p[0], &_min, &_max);
			p[0] = (p[0] - (float)_min) / ((float)_max - (float)_min);
			cv::minMaxIdx(p[1], &_min, &_max);
			p[1] = (p[1] - (float)_min) / ((float)_max - (float)_min);
			cv::minMaxIdx(p[2], &_min, &_max);
			p[2] = (p[2] - (float)_min) / ((float)_max - (float)_min);
			cv::merge(p, 3, img_to_plot);
			cv::namedWindow("blurred image", cv::WINDOW_AUTOSIZE);
			cv::imshow("blurred image", img_to_plot);
			cv::waitKey(100);
			if (param_depthdata)
			{
				cv::minMaxIdx(depth, &_min, &_max);
				img_to_plot = (depth - (float)_min) / ((float)_max - (float)_min);
				cv::namedWindow("enhanced depth", cv::WINDOW_AUTOSIZE);
				cv::imshow("enhanced depth", img_to_plot);
				cv::waitKey(100);
			}
		}

        ImageGraph G = ImageGraph(img_blurred, depth, param_pixel_vicinity, param_metrics_flag, param_z_coord_weight);
		
		n_segments = G.SegmentationKruskal(param_k);

		printf("Found segments: %7i\n", n_segments);

		G.PrintSegmentationInfo();
		
		//int *pixels_undex_thres, int *seg_under_thres, int *num_mergers
		G.Clustering(param_min_segment_size, param_target_num_segments, &pixels_under_thres, &seg_under_thres, &num_mergers);

		G.PlotSegmentation(0, "segmented filtered");
	}
	else
		print_help();

	//system("pause");
	return 0;
}
