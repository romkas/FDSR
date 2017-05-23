#include "Kruskal.h"
#include "Louvain.h"
#include "modelFitting.h"
#include "random.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>
#include <ctime>
#include <cstdio>
#include <cstring>


//void print_help();
void ReadPFMFile(cv::Mat& img, const char* filename);
//void ReadPNGFile(cv::Mat& img, const char* filename);
//int ReadImage(cv::Mat&, cv::Mat&, int, int, char**);
void NormalizeImage(cv::Mat&, cv::Mat&);
//void NormalizeDepth(cv::Mat& img);
void ScaleAndDisplay(cv::Mat &img, const char *windowname, bool needscaling, int waitkey);
void Display(cv::Mat&, const char*, int = 100);
void ResizeToCommon(cv::Mat &img, cv::Mat &depth, int wid, int hgt, bool param_depthdata);
void RunIteration(cv::Mat &img, cv::Mat &depth,	int param_pixel_vicinity, int param_edgeweight_metrics,
	float param_xy_coord_weight, float param_z_coord_weight, float param_k,
	int clustering_mode, std::vector<float>& clustering_params,
	int waitkey, const char *windowname/*, const char *logfile*/);
//void RunIterationLouvain();

SimpleGenerator RNG;
enum IMREAD_MODE
{
	read_grayscale = 1,
	read_color = 2,
	read_pfm = 4,
	read_png = 8,
	read_depth_pfm = 16,
	read_depth_png = 32,
};

int main(int argc, char **argv)
{
	cv::Mat img, img_float, img_filtered, depth, depth_float, depth_filtered;
	int c = 1;
	int param_imread_mode = std::atoi(argv[c++]);
	int param_colorspace;
	if (param_imread_mode & IMREAD_MODE::read_color)
		param_colorspace = std::atoi(argv[c++]);
	if (param_imread_mode & IMREAD_MODE::read_png)
	{
		/*if (param_imread_mode & IMREAD_MODE::read_color)
			img = cv::imread(argv[c++], cv::IMREAD_COLOR & cv::IMREAD_ANYDEPTH);
		else if (param_imread_mode & IMREAD_MODE::read_grayscale)*/
			img = cv::imread(argv[c++], cv::IMREAD_UNCHANGED);
	}
	else if (param_imread_mode & IMREAD_MODE::read_pfm)
		ReadPFMFile(img, argv[c++]);
	if (img.empty())
		return 3;
	float param_z_coord_weight;
	if (param_imread_mode & IMREAD_MODE::read_depth_pfm)
	{
		ReadPFMFile(depth, argv[c++]);
		param_z_coord_weight = std::atof(argv[c++]);
	}
	else if (param_imread_mode & IMREAD_MODE::read_depth_png)
	{
		depth = cv::imread(argv[c++], cv::IMREAD_UNCHANGED);
		if (depth.channels() > 1)
		{
			std::vector<cv::Mat> ch;
			cv::split(depth, ch);
			depth = ch[0];
		}
		param_z_coord_weight = std::atof(argv[c++]);
	}
	else
	{
		depth = cv::Mat::zeros(img.size(), CV_16UC1);
		param_z_coord_weight = 0.0f;
	}

	float param_xy_coord_weight = std::atof(argv[c++]);
	int param_pixel_vicinity = std::atoi(argv[c++]);
	int param_edgeweight_metrics = std::atoi(argv[c++]);
	float param_k = std::atof(argv[c++]);
	
	int param_min_segment_size;
	int clustering_mode = std::atoi(argv[c++]);
	if (clustering_mode & ImageGraph::REMOVE)
		param_min_segment_size = std::atoi(argv[c++]);
	
	int param_target_num_segments;

	float param_ransac_thres;
	int param_ransac_n;
	int param_ransac_d;
	int param_ransac_k;
	int param_ransac_minsegsize;

	/*(int)(std::log(1 - 0.7f) / std::log(1 - std::pow(0.8f, param_ransac_n)) +
	std::sqrt(1 - std::pow(0.8f, param_ransac_n)) / std::pow(0.8f, param_ransac_n) + 1);*/

	float param_ransacestim_regularization;
	int param_ransacestim_metrics;
	int param_modeldistance_metrics;
	float param_modeldistance_weightnormal;
	float param_modeldistance_weightdepth;
	int param_clustering_n1;
	int param_clustering_n2;
	std::vector<float> clustering_params;
	if (clustering_mode & ImageGraph::MERGE)
	{
		param_target_num_segments = std::atoi(argv[c++]);
		clustering_params.push_back(param_ransac_thres = std::atof(argv[c++]));
		clustering_params.push_back(param_ransac_n = std::atoi(argv[c++]));
		clustering_params.push_back(param_ransac_d = std::atoi(argv[c++]));
		clustering_params.push_back(param_ransac_k = std::atoi(argv[c++]));
		clustering_params.push_back(param_ransac_minsegsize = std::atoi(argv[c++]));
		clustering_params.push_back(param_ransacestim_regularization = std::atof(argv[c++]));
		clustering_params.push_back(param_ransacestim_metrics = std::atoi(argv[c++]));
		clustering_params.push_back(param_modeldistance_metrics = std::atoi(argv[c++]));
		clustering_params.push_back(param_modeldistance_weightnormal = std::atof(argv[c++]));
		clustering_params.push_back(param_modeldistance_weightdepth = std::atof(argv[c++]));
		clustering_params.push_back(param_clustering_n1 = std::atoi(argv[c++]));
		clustering_params.push_back(param_clustering_n2 = std::atoi(argv[c++]));
		clustering_params.push_back(param_target_num_segments);
	}
	if (clustering_mode & ImageGraph::REMOVE)
		clustering_params.push_back(param_min_segment_size);
	
    //char logfilename[FILENAME_MAX];
    //std::strcpy(logfilename, argv[c++]);
	//int param_segment_size_vis = std::atoi(argv[c++]);
	//bool param_color = (bool)std::atoi(argv[c++]);
		
//#if DEPTH_AS_INPUT == 1
//		ScaleAndDisplay(img, "source image", true, 100);
//#else
//		ScaleAndDisplay(img, "source image", false, 100);
//#endif
//		if (param_depthdata)
//			ScaleAndDisplay(depth, "source depth", true, 100);

	/*cv::Mat image3(img_float.size(), img_float.type());
	img_float.copyTo(image3);

	cv::GaussianBlur(image3, image3, cv::Size(5, 5), 0.7, 0.7, cv::BORDER_REFLECT101);
	ScaleAndDisplay(image3, "blurred image-3", true, 100);*/

	//cv::GaussianBlur(img_float, img_float, cv::Size(5, 5), 0.7, 0.7, cv::BORDER_CONSTANT);
	//ScaleAndDisplay(img_float, "blurred image", true, 100);
	
	//std::vector<int> v{ 1, 2 };

	NormalizeImage(img, img_float);
	NormalizeImage(depth, depth_float);

	std::chrono::high_resolution_clock localtimer;
	auto start = localtimer.now();
	if (param_colorspace == 1)
		cv::cvtColor(img_float, img_float, cv::COLOR_BGR2Lab);
	else if (param_colorspace == 2)
		cv::cvtColor(img_float, img_float, cv::COLOR_BGR2HSV);
	auto elapsed = localtimer.now() - start;
	long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("TIME (Color space conversion) (ms): %8.3f\n", (double)b / 1000);
	
	if (param_colorspace)
		NormalizeImage(img_float, img_float);


	cv::medianBlur(img_float, img_filtered, 5);
	Display(img_float, "src image");
	Display(img_filtered, "filtered image");
	Display(depth_float, "depth image");
	cv::medianBlur(depth_float, depth_filtered, 5);
	Display(depth_filtered, "depth median filter");


	RunIteration(
		img_filtered,
		depth_filtered,
		param_pixel_vicinity,
		param_edgeweight_metrics,
		param_xy_coord_weight,
		param_z_coord_weight,
		param_k,
		clustering_mode,
		clustering_params,
		0,
		"segmentation-blur");

//		cv::Mat laplacian;
//		if (param_depthdata)
//			laplacian = cv::Mat(depth.size(), depth.type());
//#if DEPTH_AS_INPUT == 1
//		laplacian = cv::Mat(img.size(), img.type());
//#endif

		//// blurred image + original depth
		//RunIteration(
		//	img_float,
		//	depth,
		//	param_pixel_vicinity,
		//	param_metrics_flag,
		//	param_z_coord_weight,
		//	param_k,
		//	param_min_segment_size,
		//	param_target_num_segments,
		//	ImageGraph::ClusteringMode::REMOVE,
		//	100,
		//	"segmentation");

		

		//if (param_depthdata)
		//{
		//	cv::GaussianBlur(depth, depth, cv::Size(5, 5), 0.7, 0.7, cv::BORDER_CONSTANT);
		//	ScaleAndDisplay(depth, "blurred depth", true, 100);

		//	/*cv::minMaxIdx(laplacian, &min, &max);
		//	laplacian = (laplacian - (float)min) / ((float)max - (float)min);

		//	int histogram[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
		//	for (int i = 0; i < laplacian.rows; i++)
		//		for (int j = 0; j < laplacian.cols; j++)
		//		{
		//			if (laplacian.at<float>(i, j) == 1)
		//				histogram[24]++;
		//			else
		//				histogram[(int)(laplacian.at<float>(i, j) * 255) / 10]++;
		//		}
		//	for (int t = 0; t < 25; t++)
		//	{
		//		printf("bin #%2i: %i\n", t + 1, histogram[t]);
		//	}*/

		//  // blurred image + blurred depth
		//	RunIteration(
		//		img_float,
		//		depth,
		//		param_pixel_vicinity,
		//		param_metrics_flag,
		//		param_z_coord_weight,
		//		param_k,
		//		param_min_segment_size,
		//		param_target_num_segments,
		//		ImageGraph::ClusteringMode::REMOVE,
		//		100,
		//		"segmentation-blur");
		//}

		

		

		//if (param_depthdata)
		//{
		//	cv::Mat depth0(depth.size(), depth.type());
			
			/*cv::Laplacian(depth, laplacian, laplacian.depth(), 3, 1., 0., cv::BORDER_REFLECT101);
			cv::minMaxIdx(laplacian, &min, &max);
			laplacian = (laplacian - (float)min) / ((float)max - (float)min);
			ScaleAndDisplay(laplacian, "laplacian", true, 100);*/

		//	cv::Mat depth_blur(depth.size(), depth.type());
			//cv::GaussianBlur(depth, depth_blur, cv::Size(3, 3), 0.7, 0.7, cv::BORDER_CONSTANT);
			
			//ScaleAndDisplay(depth_blur, "blurred depth", true, 100);
			
		//	depth_blur.copyTo(depth0);

			/*cv::minMaxIdx(depth, &min, &max);
			depth = (depth - (float)min) / ((float)max - (float)min);
			depth += laplacian;
			cv::minMaxIdx(depth, &min, &max);
			depth = (depth - (float)min) / ((float)max - (float)min);
			ScaleAndDisplay(depth, "depth + laplacian", false, 100);

			cv::minMaxIdx(depth_blur, &min, &max);
			depth_blur = (depth_blur - (float)min) / ((float)max - (float)min);
			depth_blur += laplacian;
			cv::minMaxIdx(depth_blur, &min, &max);
			depth_blur = (depth_blur - (float)min) / ((float)max - (float)min);
			ScaleAndDisplay(depth_blur, "depth-blur + laplacian", false, 100);*/

            //double min11, max11;
			//double min12, max12;
			//double min13, max13;

            //cv::minMaxIdx(depth0, &q1, &q2);
			//depth0 = (depth0 - (float)q1) / ((float)q2 - (float)q1);
			
			/*cv::Mat channels[3];
			cv::split(depth, channels);
			cv::minMaxIdx(channels[0], &min11, &max11);
			cv::minMaxIdx(channels[1], &min12, &max12);
			cv::minMaxIdx(channels[2], &min13, &max13);*/
			

			


			// blurred image + blurred depth
			/*RunIteration(
				img_float,
				depth,
				param_pixel_vicinity,
				param_edgeweight_metrics,
				param_xy_coord_weight,
				param_z_coord_weight,
				param_k,
				param_min_segment_size,
				param_target_num_segments,
				clustering_mode,
				clustering_params,
				100,
				"segmentation-blur");*/
		//}
		//else
		//{
		//	// given depth as image, combine the image and its laplacian
		//	cv::Laplacian(img_float, laplacian, laplacian.depth(), 3, 1., 0., cv::BORDER_REFLECT101);
		//	cv::minMaxIdx(laplacian, &min, &max);
		//	laplacian = (laplacian - (float)min) / ((float)max - (float)min);
		//	cv::minMaxIdx(img_float, &min, &max);
		//	img_float = (img_float - (float)min) / ((float)max - (float)min);
		//	img_float += laplacian;
		//	cv::minMaxIdx(img_float, &min, &max);
		//	img_float = (img_float - (float)min) / ((float)max - (float)min);
		//	ScaleAndDisplay(img_float, "depth + laplacian", false, 100);

		//	RunIteration(
		//		img_float,
		//		depth,
		//		param_pixel_vicinity,
		//		param_edgeweight_metrics,
		//		param_xy_coord_weight,
		//		param_z_coord_weight,
		//		param_k,
		//		param_min_segment_size,
		//		param_target_num_segments,
		//		clustering_mode,
		//		clustering_params,
		//		100,
		//		"segmentation-blur-laplace");
		//}

		//cv::waitKey();
	//}
	//else
	//	print_help();

	//system("pause");

    //SimpleGenerator::Release();

	return 0;
}

//void print_help()
//{
//	printf("Program usage:\narg1 - size of pixel vicinity (4, 8, 24, 48)\narg2 - metric function for edge weight calculation\n"
//		"arg3 - Kruskal k parameter\narg4 - segment size threshold\narg5 - target number of segments\n"
//		"arg6 - z");
//#if USE_COLOR == 1
//	printf("color image file path");
//#else
//	printf("grayscale image file path");
//#endif
//	printf("\narg7 - is depth map data given\narg8 - z coordinate scaling\narg9 - depth data file path");
//}

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

void NormalizeImage(cv::Mat & img, cv::Mat & img_float)
{
	double min, max;
	if (img.channels() == 1)
	{
		cv::minMaxIdx(img, &min, &max);
		if (min == max)
			img.convertTo(img_float, CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
		else
			img.convertTo(img_float, CV_32FC1, 1.0 / (max - min), -min / (max - min));
	}
	else
	{
		cv::Mat channels[3];
		cv::split(img, channels);
		cv::minMaxIdx(channels[0], &min, &max);
		if (min == max)
			channels[0].convertTo(channels[0], CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
		else
			channels[0].convertTo(channels[0], CV_32FC1, 1.0 / (max - min), -min / (max - min));
		cv::minMaxIdx(channels[1], &min, &max);
		if (min == max)
			channels[1].convertTo(channels[1], CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
		else
			channels[1].convertTo(channels[1], CV_32FC1, 1.0 / (max - min), -min / (max - min));
		cv::minMaxIdx(channels[2], &min, &max);
		if (min == max)
			channels[2].convertTo(channels[2], CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
		else
			channels[2].convertTo(channels[2], CV_32FC1, 1.0 / (max - min), -min / (max - min));
		cv::merge(channels, 3, img_float);
	}
}

//void NormalizeDepth(cv::Mat & img)
//{
//	int k = img.depth() == CV_8U ? 255 : (img.depth() == CV_32F ? 65535 : 1);
//	double min;
//	if (img.channels() == 1)
//	{
//		cv::minMaxIdx(img, &min);
//		img.convertTo(img, CV_16UC1, (double)k, min < 0 ? -min * k : 0.0);
//	}
//	else
//	{
//		cv::Mat channels[3];
//		cv::split(img, channels);
//		cv::minMaxIdx(channels[0], &min);
//		channels[0].convertTo(img, CV_16UC1, (double)k, min < 0 ? -min * k : 0.0);
//	}
//}

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

void Display(cv::Mat &m, const char *wname, int key)
{
	cv::namedWindow(wname, cv::WINDOW_AUTOSIZE);
	cv::imshow(wname, m);
	cv::waitKey(key);
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
	int param_edgeweight_metrics,
	float param_xy_coord_weight,
	float param_z_coord_weight,
	float param_k,
	int clustering_mode,
	std::vector<float>& clustering_params,
	int waitkey,
	const char *windowname/*,
	const char *logfile*/)
{
	int n_segments;
	int pixels_under_thres, seg_under_thres, num_mergers;
	float totalerror;
	printf("==================================\n");
	ImageGraph G = ImageGraph(
		img,
		depth,
		param_pixel_vicinity,
		param_edgeweight_metrics,
		param_xy_coord_weight,
		param_z_coord_weight);
	n_segments = G.SegmentationKruskal(param_k);
	G.Refine(
		clustering_mode,
		clustering_params,
		&pixels_under_thres, &seg_under_thres, &num_mergers, &totalerror);
	printf("Found segments:           %7i\n", n_segments);
	printf("Pixels under threshold:   %7i\nSegments under threshold: %7i\nNumber of mergers:        %7i\n", pixels_under_thres, seg_under_thres, num_mergers);
	printf("RANSAC total error: %e\n", totalerror);
	//G.PrintSegmentationInfo(logfile);
	printf("==================================\n");
	G.PlotSegmentation(waitkey, windowname);
}
