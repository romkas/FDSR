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

void ScaleAndDisplay(cv::Mat &img, const char *windowname, bool needscaling, int waitkey)
{
	double min, max;
	cv::Mat img_displayed;
	if (needscaling)
	{
		img_displayed = cv::Mat(img.size(), img.type());
		if (img.channels() == 1)
		{
			cv::minMaxIdx(img, &min, &max);
			img_displayed = (img - (float)min) / ((float)max - (float)min);
		}
		else
		{
			cv::Mat channels[3];
			cv::split(img, channels);
			cv::minMaxIdx(channels[0], &min, &max);
			channels[0] = (channels[0] - (float)min) / ((float)max - (float)min);
			cv::minMaxIdx(channels[1], &min, &max);
			channels[1] = (channels[1] - (float)min) / ((float)max - (float)min);
			cv::minMaxIdx(channels[2], &min, &max);
			channels[2] = (channels[2] - (float)min) / ((float)max - (float)min);
			cv::merge(channels, 3, img_displayed);
		}
	}
	else
		img_displayed = img;
	cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowname, img_displayed);
	cv::waitKey(waitkey);
}

void ResizeToCommon(cv::Mat &img, cv::Mat &depth, int wid, int hgt, bool param_depthdata)
{
	int w, h;
	cv::Size imsz(img.size());
	if (wid == 0 && hgt == 0)
	{
		cv::Size depsz(depth.size());
		w = cv::min(imsz.width, depsz.width);
		h = cv::max(imsz.height, depsz.height);
	}
	else
	{
		w = wid;
		h = hgt;
	}
	cv::Size targetsize(w, h);
	cv::resize(img, img, targetsize, 0., 0., cv::INTER_LINEAR);
	if (param_depthdata)
		cv::resize(depth, depth, targetsize, 0., 0., cv::INTER_LINEAR);
}

void RunIteration(
	cv::Mat &img,
	cv::Mat &depth,
	int param_pixel_vicinity,
	int param_metrics_flag,
	double param_z_coord_weight,
	double param_k,
	int param_min_segment_size,
	int param_target_num_segments,
	int waitkey,
	const char *windowname)
{
	int n_segments;
	int pixels_under_thres, seg_under_thres, num_mergers;
	printf("==================================\n");
	ImageGraph G = ImageGraph(img, depth, param_pixel_vicinity, param_metrics_flag, param_z_coord_weight);
	n_segments = G.SegmentationKruskal(param_k);
	printf("Found segments: %7i\n", n_segments);
	G.Clustering(param_min_segment_size, param_target_num_segments, &pixels_under_thres, &seg_under_thres, &num_mergers);
	G.PlotSegmentation(waitkey, windowname);
	printf("==================================\n");
}

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
#elif DEPTH_AS_INPUT
		ReadPFMFile(img, argv[c++]);
#else
		img = cv::imread(argv[c++], cv::IMREAD_GRAYSCALE);
#endif

		double _min, _max;

		bool param_depthdata = (bool)std::atoi(argv[c++]);
		if (param_depthdata)
		{
			ReadPFMFile(depth, argv[c++]);
			param_z_coord_weight = std::atof(argv[c++]);
			ResizeToCommon(img, depth, 320, 240, true);
		}
		else
		{
			ResizeToCommon(img, depth, 320, 240, false);
			depth = cv::Mat::zeros(img.size(), img.type());
			param_z_coord_weight = 1.;
		}

		if (img.depth() != CV_32F)
#if USE_COLOR == 1
			img.convertTo(img_float, CV_32FC3);
#else
			img.convertTo(img_float, CV_32FC1);
#endif
		else
			img_float = img;

#if DEPTH_AS_INPUT == 1
		ScaleAndDisplay(img, "source image", true, 100);
#else
		ScaleAndDisplay(img, "source image", false, 100);
#endif
		if (param_depthdata)
			ScaleAndDisplay(depth, "source depth", true, 100);

		cv::GaussianBlur(img_float, img_float, cv::Size(5, 5), 0.7, 0.7, cv::BORDER_CONSTANT);
		ScaleAndDisplay(img_float, "blurred image", true, 100);

		cv::Mat laplacian;
		if (param_depthdata)
			laplacian = cv::Mat(depth.size(), depth.type());
#if DEPTH_AS_INPUT == 1
		laplacian = cv::Mat(img.size(), img.type());
#endif

		RunIteration(
			img_float,
			depth,
			param_pixel_vicinity,
			param_metrics_flag,
			param_z_coord_weight,
			param_k,
			param_min_segment_size,
			param_target_num_segments,
			100,
			"segmentation-1");

		double min, max;

		if (param_depthdata)
		{
			cv::GaussianBlur(depth, depth, cv::Size(5, 5), 0.7, 0.7, cv::BORDER_CONSTANT);
			ScaleAndDisplay(depth, "blurred depth", true, 100);

			cv::minMaxIdx(laplacian, &min, &max);
			laplacian = (laplacian - (float)min) / ((float)max - (float)min);

			int histogram[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
			for (int i = 0; i < laplacian.rows; i++)
				for (int j = 0; j < laplacian.cols; j++)
				{
					if (laplacian.at<float>(i, j) == 1)
						histogram[24]++;
					else
						histogram[(int)(laplacian.at<float>(i, j) * 255) / 10]++;
				}
			for (int t = 0; t < 25; t++)
			{
				printf("bin #%2i: %i\n", t + 1, histogram[t]);
			}

			RunIteration(
				img_float,
				depth,
				param_pixel_vicinity,
				param_metrics_flag,
				param_z_coord_weight,
				param_k,
				param_min_segment_size,
				param_target_num_segments,
				100,
				"segmentation-2");
		}

		if (param_depthdata)
		{
			cv::Laplacian(depth, laplacian, laplacian.depth(), 3, 1., 0., cv::BORDER_REFLECT101);
			cv::minMaxIdx(laplacian, &min, &max);
			laplacian = (laplacian - (float)min) / ((float)max - (float)min);
			cv::minMaxIdx(depth, &min, &max);
			depth = (depth - (float)min) / ((float)max - (float)min);
			depth += laplacian;
			cv::minMaxIdx(depth, &min, &max);
			depth = (depth - (float)min) / ((float)max - (float)min);
			ScaleAndDisplay(depth, "depth + laplacian", false, 100);

			RunIteration(
				img_float,
				depth,
				param_pixel_vicinity,
				param_metrics_flag,
				param_z_coord_weight,
				param_k,
				param_min_segment_size,
				param_target_num_segments,
				100,
				"segmentation-3");
		}
		else
		{
			cv::Laplacian(img_float, laplacian, laplacian.depth(), 3, 1., 0., cv::BORDER_REFLECT101);
			cv::minMaxIdx(laplacian, &min, &max);
			laplacian = (laplacian - (float)min) / ((float)max - (float)min);
			cv::minMaxIdx(img_float, &min, &max);
			img_float = (img_float - (float)min) / ((float)max - (float)min);
			img_float += laplacian;
			cv::minMaxIdx(img_float, &min, &max);
			img_float = (img_float - (float)min) / ((float)max - (float)min);
			ScaleAndDisplay(img_float, "depth + laplacian", false, 100);

			RunIteration(
				img_float,
				depth,
				param_pixel_vicinity,
				param_metrics_flag,
				param_z_coord_weight,
				param_k,
				param_min_segment_size,
				param_target_num_segments,
				100,
				"segmentation-2");
		}

		cv::waitKey();
	}
	else
		print_help();

	//system("pause");
	return 0;
}
