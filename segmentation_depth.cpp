#include "disjointSetClass.h"
#include "Kruskal.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
//#include <fstream>
#include <cstring>

using namespace std;

void print_help()
{
	printf("Program usage: ...\n");
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



// argv[1] - image file
// argv[2] - size of pixel vicinity
// argv[3] - either use pixel distance or not
// argv[4] - min segment size
// argv[5] - Kruskal k parameter
int main(int argc, char **argv)
{
	if (argc >= 6)
	{
		cv::Mat img, img_to_plot;
		ReadPFMFile(img, argv[1]);

		cv::Size img_size = img.size();
		int width = img_size.width, height = img_size.height;
		int img_type = img.type();

		double min, max;
		cv::minMaxIdx(img, &min, &max);
		img_to_plot = (img - (float)min) / ((float)max - (float)min);
		cv::namedWindow("loaded image", cv::WINDOW_AUTOSIZE);
		cv::imshow("loaded image", img_to_plot);
		cv::waitKey(1000);
		
        int pixel_vicinity = std::atoi(argv[2]);
        bool use_pixel_distance = (bool)std::atoi(argv[3]);
        int min_segment_size = std::atoi(argv[4]);
        int kruskal_k_param = std::atoi(argv[5]);

        //DisjointSet<float> djset;
        //graph::ImageGraph<float> G;
        ImageGraph G = ImageGraph(img, use_pixel_distance , pixel_vicinity);
		cv::Mat labels = -cv::Mat::ones(img_size, CV_32SC1);
		int n_segments;
		n_segments = G.SegmentationKruskal(labels, min_segment_size, kruskal_k_param);
		
		//printf("Found segments: %4i\n", n_segments);

        {
            cv::Mat p[3];
			cv::Mat img_3channel(img_size, CV_32FC3);
            img_to_plot.copyTo(p[0]);
			img_to_plot.copyTo(p[1]);
			img_to_plot.copyTo(p[2]);
			p[0].convertTo(p[0], CV_8UC1, 255.);
			p[1].convertTo(p[1], CV_8UC1, 255.);
			p[2].convertTo(p[2], CV_8UC1, 255.);

			// paint pixels according to segment labels
			std::vector<int> segment_labels;
			std::vector<unsigned char> red, green, blue;
			int pos_in_color_vector;
			cv::RNG rng;
			for (int i = 0; i < img_3channel.rows; i++)
				for (int j = 0; j < img_3channel.cols; j++)
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
						p[0].at<unsigned char>(i, j) = red[pos_in_color_vector];
						p[1].at<unsigned char>(i, j) = green[pos_in_color_vector];
						p[2].at<unsigned char>(i, j) = blue[pos_in_color_vector];
					}
				}

			cv::merge(p, 3, img_3channel);
			cv::namedWindow("segmented image", cv::WINDOW_AUTOSIZE);
			cv::imshow("segmented image", img_3channel);
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
