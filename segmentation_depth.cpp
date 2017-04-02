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
	FILE *stream = fopen(filename, "rb");
	if (stream == 0)
		cout << "ReadFile: could not open %s" << endl;

	int width, height;
	float scale;
	int im_type;
	float tmp;
	char s[5];

	fread(s, sizeof(char), 2, stream);
	s[2] = '\0';
	im_type = (strcmp(s, "Pf") == 0) ? CV_32FC1 : CV_32FC3;

	fread(s, sizeof(char), 1, stream);
	fread(s, sizeof(char), 3, stream);
	s[3] = '\0';
	width = atoi(s);

	fread(s, sizeof(char), 1, stream);

	fread(s, sizeof(char), 3, stream);
	s[3] = '\0';
	height = atoi(s);

	fread(s, sizeof(char), 1, stream);
	fread(s, sizeof(char), 3, stream);
	s[2] = '\0';
	scale = atof(s);

	img = cv::Mat(cv::Size(width, height), im_type);

	for (int y = height - 1; y >= 0; y--)
	{
		for (int x = 0; x < width; x++)
		{
			if ((int)fread(&tmp, sizeof(float), 1, stream) != 1)
				cout << "ReadFile(%s): file is too short" << endl;
			s[4] = '\0';

			img.at<float>(y, x) = -tmp;
			//img.at<float>(y, x) = -atof(s);
		}

	}

	if (fgetc(stream) != EOF)
		cout << "ReadFile(%s): file is too long" << endl;

	fclose(stream);
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
		cv::Mat input_img;
		//input_img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		ReadPFMFile(input_img, argv[1]);

		cv::Size img_size = input_img.size();
		int width = img_size.width, height = img_size.height;
		int img_type = input_img.type();

		cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_image", input_img);

		graph::ImageGraph<float> G(input_img, 1, 4);
		cv::Mat labels = -cv::Mat::ones(img_size, img_type);
		int n_segments;
		n_segments = G.SegmentationKruskal(labels, 0, 0);


	}
	else
		print_help();

	//system("pause");
	return 0;
}
