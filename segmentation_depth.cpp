#include "disjointSetClass.h"
#include "Kruskal.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <iostream>
//#include <fstream>
#include <cstring>

using namespace std;

void print_help()
{
	printf("Program usage:\narg1 - size of pixel vicinity (4, 8, 24, 48)\narg2 - metric function for edge weight calculation\n"
		"arg3 - Kruskal k parameter\narg4 - segment size threshold\narg5 - target number of segments\n"
		"arg6 - segment size threshold (for visualization)\narg7 - ");
#if USE_COLOR == 1
	printf("color image file path");
#else
	printf("grayscale image file path");
#endif
	printf("\narg8 - is depth map data given\narg9- depth data file path");
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
	if (argc >= 9)
	{
		int c = 1;
		int param_pixel_vicinity = std::atoi(argv[c++]);
		int param_metrics_flag = std::atoi(argv[c++]);
		double param_k = std::atof(argv[c++]);
		int param_min_segment_size = std::atoi(argv[c++]);
		int param_target_num_segments = std::atoi(argv[c++]);
		int param_segment_size_vis = std::atoi(argv[c++]);
		//bool param_color = (bool)std::atoi(argv[c++]);
		double param_z_coord_weight = std::atof(argv[c++]);

		cv::Mat img, img_float, img_to_plot, depth;
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

		bool param_depthdata = (bool)std::atoi(argv[c++]);
		if (param_depthdata)
			ReadPFMFile(depth, argv[c++]);
		else
			depth = cv::Mat::zeros(img_size, img_type);

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

		{
			double min, max;
			/*img.convertTo(img_to_plot, CV_8UC3, 255.);
			cv::Mat p[3];
			cv::split(img_to_plot, p);
			cv::minMaxIdx(p[0], &min, &max);
			p[0] = (p[0] - (float)min) / ((float)max - (float)min);
			cv::minMaxIdx(p[1], &min, &max);
			p[1] = (p[1] - (float)min) / ((float)max - (float)min);
			cv::minMaxIdx(p[2], &min, &max);
			p[2] = (p[2] - (float)min) / ((float)max - (float)min);
			cv::merge(p, 3, img_to_plot);*/
			img_to_plot = cv::Mat(img);
			cv::namedWindow("source image", cv::WINDOW_AUTOSIZE);
			cv::imshow("source image", img_to_plot);
			cv::waitKey(100);
			if (param_depthdata)
			{
				cv::minMaxIdx(depth, &min, &max);
				img_to_plot = (depth - (float)min) / ((float)max - (float)min);
				cv::namedWindow("source depth", cv::WINDOW_AUTOSIZE);
				cv::imshow("source depth", img_to_plot);
				cv::waitKey(100);
			}
		}

        ImageGraph G = ImageGraph(img_float, depth, param_pixel_vicinity, param_metrics_flag, param_z_coord_weight);
		cv::Mat labels = -cv::Mat::ones(img_size, CV_32SC1);
		//int n_segments;
		G.SegmentationKruskal(labels, param_k, param_min_segment_size, param_segment_size_vis, param_target_num_segments);
		
		//printf("Found segments: %4i\n", n_segments);

        {
			cv::Mat segmentation = cv::Mat::zeros(img_size, CV_8UC3);

			/*cv::Mat p[3];
			cv::Mat img_3channel(img_size, CV_32FC3);
            img_to_plot.copyTo(p[0]);
			img_to_plot.copyTo(p[1]);
			img_to_plot.copyTo(p[2]);
			p[0].convertTo(p[0], CV_8UC1, 255.);
			p[1].convertTo(p[1], CV_8UC1, 255.);
			p[2].convertTo(p[2], CV_8UC1, 255.);*/

			// paint pixels according to segment labels
			std::vector<int> segment_labels;
			std::vector<unsigned char> red, green, blue;
			int pos_in_color_vector;
			cv::RNG rng;
			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
				{
					pos_in_color_vector = -1;
					for (int t = 0; t < segment_labels.size(); t++)
					{
						if (labels.at<int>(i, j) == segment_labels[t])
						{
							pos_in_color_vector = t;
							break;
						}
					}
					if (labels.at<int>(i, j) != -1 && pos_in_color_vector == -1)
					{
						// generate new color for given segment
						int a = 120, b = 256;
						red.push_back(rng.uniform(a, b));
						green.push_back(rng.uniform(a, b));
						blue.push_back(rng.uniform(a, b));
						segment_labels.push_back(labels.at<int>(i, j));
						pos_in_color_vector = segment_labels.size() - 1;
					}
					else
					{
						// use one of the colors calculated before
					}
					if (pos_in_color_vector != -1)
					{
						/*p[0].at<unsigned char>(i, j) = red[pos_in_color_vector];
						p[1].at<unsigned char>(i, j) = green[pos_in_color_vector];
						p[2].at<unsigned char>(i, j) = blue[pos_in_color_vector];*/
						segmentation.at<cv::Vec3b>(i, j) = cv::Vec3b(red[pos_in_color_vector],
							green[pos_in_color_vector], blue[pos_in_color_vector]);
					}
				}

			//cv::merge(p, 3, img_3channel);
			cv::namedWindow("segmented image", cv::WINDOW_AUTOSIZE);
			//cv::imshow("segmented image", img_3channel);
			cv::imshow("segmented image", segmentation);
			cv::waitKey();

            /*cv::Mat r(img_size, CV_32FC1), g(img_size, CV_32FC1), b(img_size, CV_32FC1);
            cv::extractChannel(img, r, 1);
            cv::extractChannel(img, g, 2);
            cv::extractChannel(img, b, 3);
            cv::minMaxIdx(r, &min, &max);
            r = (r - (float)min) / ((float)max - (float)min);
            cv::minMaxIdx(g, &min, &max);
            g = (g - (float)min) / ((float)max - (float)min);
            cv::minMaxIdx(b, &min, &max);
            b = (b - (float)min) / ((float)max - (float)min);
            p[0] = r;
            p[1] = g;
            p[2] = b;
            cv::merge(p, 3, img_to_plot);

            cv::namedWindow("rgb image", cv::WINDOW_AUTOSIZE);
            cv::imshow("rgb image", img_to_plot);
            cv::waitKey();*/


        }
	}
	else
		print_help();

	//system("pause");
	return 0;
}
