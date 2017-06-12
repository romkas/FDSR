#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <map>
#include "random.h"
//#include "noname.h"

#if (RUN == 1 || RUN == 2)
#include "evaluate.h"
#include "util.h"
#include <iterator>
#include <map>
#include <algorithm>
#include <utility>
#include <cstdio>
#include <cstring>
#endif


using namespace std;


SimpleGenerator RNG;
int param_verbosity;
long long global_counter;
long long counter1, counter2, counter3, counter4;

//const char* const PLANES_FILE = "F:\\opticalflow\\tests\\planes.txt";
//const char* const RANSAC_FILE = "F:\\opticalflow\\tests\\ransac.txt";
//const char* const RT_FILE = "F:\\opticalflow\\tests\\runtime.txt";


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

#if (RUN == 1 || RUN == 2)

//void print_help();
void ReadPFMFile(cv::Mat& img, const char* filename);
//void ReadPNGFile(cv::Mat& img, const char* filename);
//int ReadImage(cv::Mat&, cv::Mat&, int, int, char**);
//void NormalizeImage(cv::Mat&, cv::Mat&);
//void NormalizeDepth(cv::Mat& img);
void ScaleAndDisplay(cv::Mat &img, const char *windowname, bool needscaling, int waitkey);
void Display(cv::Mat&, const char*, int = 100);
void ResizeToCommon(cv::Mat &img, cv::Mat &depth, int wid, int hgt, bool param_depthdata);
void PlotSegmentation(cv::Mat&, int, const char*);
void PartitionStatistics(vector<list<cv::Vec2i>>&, double*, double*);

void ReadLabelsFile(cv::Mat&, const char*);


enum IMREAD_MODE
{
	read_grayscale = 1,
	read_color = 2,
	read_pfm = 4,
	read_png = 8,
	read_depth_pfm = 16,
	read_depth_png = 32,
	read_gt = 64,
	read_file = 128
};

int main(int argc, char **argv)
{
	std::chrono::high_resolution_clock localtimer;
	global_counter = 0;

	char buf[200];

	std::vector<std::list<cv::Vec2i>> partition0, partition, partition_badmodel, partition_nomerge;

	cv::Mat img, img_float, img_filtered, depth, depth_float, depth_filtered, gt;
	int c = 1;
	int param_img_need_filter;
	int param_depth_need_filter;
	int param_imread_mode = std::atoi(argv[c++]);
	int param_colorspace;

	char images_path[100], depths_path[100], gts_path[100];
	FILE *images, *depths, *gts;

	if (param_imread_mode & IMREAD_MODE::read_color)
		param_colorspace = std::atoi(argv[c++]);
	if (param_imread_mode & IMREAD_MODE::read_file)
	{
		strcpy(images_path, argv[c++]);
	}
	else if (param_imread_mode & IMREAD_MODE::read_png)
	{
		/*if (param_imread_mode & IMREAD_MODE::read_color)
			img = cv::imread(argv[c++], cv::IMREAD_COLOR & cv::IMREAD_ANYDEPTH);
		else if (param_imread_mode & IMREAD_MODE::read_grayscale)*/
		img = cv::imread(argv[c++], cv::IMREAD_UNCHANGED);
	}
	else if (param_imread_mode & IMREAD_MODE::read_pfm)
		ReadPFMFile(img, argv[c++]);
	
	if (!(param_imread_mode & IMREAD_MODE::read_file))
		if (img.empty())
			return 3;
	
	param_img_need_filter = std::atoi(argv[c++]);
	int param_filter_kernel_size;
	if (param_img_need_filter)
		param_filter_kernel_size = std::atoi(argv[c++]);
	double param_z_coord_weight;
	if (param_imread_mode & IMREAD_MODE::read_file)
	{
		strcpy(depths_path, argv[c++]);
		param_depth_need_filter = std::atoi(argv[c++]);
		param_z_coord_weight = std::atof(argv[c++]);
	}
	else if (param_imread_mode & IMREAD_MODE::read_depth_pfm)
	{
		ReadPFMFile(depth, argv[c++]);
		param_depth_need_filter = std::atoi(argv[c++]);
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
		param_depth_need_filter = std::atoi(argv[c++]);
		param_z_coord_weight = std::atof(argv[c++]);
	}
	else
	{
		depth = cv::Mat::zeros(img.size(), CV_16UC1);
		param_z_coord_weight = 0.0f;
	}
	
	int param_iter_test;
	
	if (param_imread_mode & IMREAD_MODE::read_file)
	{
		strcpy(gts_path, argv[c++]);
		param_iter_test = std::atoi(argv[c++]);
	}
	else if (param_imread_mode & IMREAD_MODE::read_gt)
	{
		ReadLabelsFile(gt, argv[c++]);
		param_iter_test = std::atoi(argv[c++]);
	}

	int param_pixel_vicinity = std::atoi(argv[c++]);
	int param_min_segment_size = std::atoi(argv[c++]);

	vector<double> alg_params;
#if SEG_ALG == 0
	double param_k;

	/*if (clustering_mode & REMOVE)
	param_min_segment_size = std::atoi(argv[c++]);*/

	alg_params.push_back(param_k = std::atof(argv[c++]));
#elif SEG_ALG == 1
	double param_min_modularity;
	int param_num_pass;
	int param_outer_iterations;

	alg_params.push_back(param_min_modularity = std::atof(argv[c++]));
	alg_params.push_back(param_num_pass = std::atoi(argv[c++]));
	alg_params.push_back(param_outer_iterations = std::atoi(argv[c++]));
#endif

	int clustering_mode = std::atoi(argv[c++]);

	int param_ransac_n;
	int param_ransac_k;
	double param_ransac_thres;
	double param_ransac_d;
	//int param_ransac_minsegsize;

	/*(int)(std::log(1 - 0.7f) / std::log(1 - std::pow(0.8f, param_ransac_n)) +
	std::sqrt(1 - std::pow(0.8f, param_ransac_n)) / std::pow(0.8f, param_ransac_n) + 1);*/

	double param_ransacestim_regularization;
	//int param_ransacestim_metrics;

	int param_target_num_segments;
	//int param_modeldistance_metrics;
	double param_modeldistance_weight;
	//double param_modeldistance_weightdepth;
	int param_clustering_n1;
	int param_clustering_n2;

	std::vector<double> clustering_params, modelling_params;
	if (clustering_mode & MODE_MERGE)
	{
		modelling_params.push_back(param_ransac_n = std::atoi(argv[c++]));
		modelling_params.push_back(param_ransac_k = std::atoi(argv[c++]));
		modelling_params.push_back(param_ransac_thres = std::atof(argv[c++]));
		modelling_params.push_back(param_ransac_d = std::atof(argv[c++]));
		/*modelling_params.push_back(param_ransac_minsegsize = std::atoi(argv[c++]));*/
		modelling_params.push_back(param_ransacestim_regularization = std::atof(argv[c++]));
		//modelling_params.push_back(param_ransacestim_metrics = std::atoi(argv[c++]));
		clustering_params.push_back(param_target_num_segments = std::atoi(argv[c++]));
		//clustering_params.push_back(param_modeldistance_metrics = std::atoi(argv[c++]));
		clustering_params.push_back(param_modeldistance_weight = std::atof(argv[c++]));
		/*clustering_params.push_back(param_modeldistance_weightdepth = std::atof(argv[c++]));*/
		clustering_params.push_back(param_clustering_n1 = std::atoi(argv[c++]));
		clustering_params.push_back(param_clustering_n2 = std::atoi(argv[c++]));
	}
	/*if (clustering_mode & REMOVE)
		clustering_params.push_back(param_min_segment_size);*/

	//char param_folder[100];
	//if (param_imread_mode & IMREAD_MODE::read_file)
	//	strcpy(param_folder, argv[c++]);

//#if RUN == 2
//	char folder[100];
//	strcpy(folder, argv[c++]);
//#endif
	param_verbosity = std::atoi(argv[c++]);

	//std::vector<std::list<cv::Vec2i>> partition0, partition, partition_badmodel, partition_nomerge;
	if (param_imread_mode & IMREAD_MODE::read_file)
	{
		images = fopen(images_path, "r");
		//if (param_imread_mode & (IMREAD_MODE::read_depth_png | IMREAD_MODE::read_depth_pfm))
		depths = fopen(depths_path, "r");
		gts = fopen(gts_path, "r");

		cv::Mat colors;
		cv::Mat labels;
		const char image_dir[] = "\\_gt_test\\";
		

		

		while (fscanf(images, "%s\n", buf) != EOF)
		{
			img = cv::imread(buf, cv::IMREAD_UNCHANGED);
			fscanf(depths, "%s\n", buf);
			depth = cv::imread(buf, cv::IMREAD_UNCHANGED);
			if (depth.channels() > 1)
			{
				std::vector<cv::Mat> ch;
				cv::split(depth, ch);
				depth = ch[0];
			}

			gt = cv::Mat(img.size(), CV_32SC1);
			fscanf(gts, "%s\n", buf);
			ReadLabelsFile(gt, buf);

			NormalizeImage(img, img_float);
			NormalizeImage(depth, depth_float);

			//auto start = localtimer.now();
			if (param_colorspace == 1)
				cv::cvtColor(img_float, img_float, cv::COLOR_BGR2Lab);
			else if (param_colorspace == 2)
				cv::cvtColor(img_float, img_float, cv::COLOR_BGR2HSV);
			//auto elapsed = localtimer.now() - start;
			//long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			//if (param_verbosity > 0)
			//	printf("TIME (Color space conversion) (ms): %8.3f\n", (double)b / 1000);
			//FILE *rt = fopen(RT_FILE, "a");
			//fprintf(rt, "%8.3f\n", (double)b / 1000);
			//fclose(rt);
			//global_counter += b;

			if (param_colorspace)
				NormalizeImage(img_float, img_float);

			if (param_img_need_filter)
			{
				//start = localtimer.now();

				cv::medianBlur(img_float, img_filtered, param_filter_kernel_size);

				//elapsed = localtimer.now() - start;
				//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
				//if (param_verbosity > 0)
				//	printf("TIME (Apply filter to image) (ms): %8.3f\n", (double)b / 1000);

				//rt = fopen(RT_FILE, "a");
				//fprintf(rt, "%8.3f\n", (double)b / 1000);
				//fclose(rt);

				//global_counter += b;
			}
			else
				img_filtered = img_float;
			if (param_depth_need_filter)
			{
				//start = localtimer.now();

				cv::medianBlur(depth_float, depth_filtered, param_filter_kernel_size);

				//elapsed = localtimer.now() - start;
				//b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
				//if (param_verbosity > 0)
				//	printf("TIME (Apply filter to depth) (ms): %8.3f\n", (double)b / 1000);

				//rt = fopen(RT_FILE, "a");
				//fprintf(rt, "%8.3f\n", (double)b / 1000);
				//fclose(rt);

				//global_counter += b;
			}
			else
				depth_filtered = depth_float;

			
			//std::vector<std::list<cv::Vec2i>> partition_nomerge(n_segments);

			if (param_imread_mode & IMREAD_MODE::read_gt)
			{
				if (param_iter_test > 1)
				{
					std::vector<double> graph_params{ (double)param_pixel_vicinity, param_z_coord_weight };
					eval::TestAlgorithm(img_filtered, depth_filtered, gt, graph_params, param_min_segment_size,
						alg_params, modelling_params, clustering_params, clustering_mode,
						partition0, partition, partition_nomerge, partition_badmodel, param_iter_test);
				}
				else {}
			}
			
			char *P = strrchr(buf, '\\');
			char name[200];
			char temp[200];
			strcpy(name, P + 1);
			strcpy(P, image_dir);
			P = strrchr(name, '.');
			strcpy(P, ".pcd");
			strcat(buf, name);
			strcpy(temp, buf);

			colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
			labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
			LabelPartition(partition0, labels, colors);
			cv::imwrite(strcat(buf, "\\initial.png"), colors);
			if (clustering_mode & MODE_REMOVE)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_nomerge, labels, colors);
				cv::imwrite(strcat(strcpy(buf, temp), "\\no_merge.png"), colors);
			}
			if (clustering_mode & MODE_MERGE)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition, labels, colors);
				cv::imwrite(strcat(strcpy(buf, temp), "\\hac.png"), colors);
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_badmodel, labels, colors);
				cv::imwrite(strcat(strcpy(buf, temp), "\\invalid.png"), colors);
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				partition.insert(partition.end(), partition_badmodel.begin(), partition_badmodel.end());
				LabelPartition(partition, labels, colors);
				cv::imwrite(strcat(strcpy(buf, temp), "\\result.png"), colors);
			}

			partition0.clear();
			partition.clear();
			partition_badmodel.clear();
			partition_nomerge.clear();
		}
		
		fclose(images);
		fclose(depths);
		fclose(gts);
	}
	else
	{
		NormalizeImage(img, img_float);
		NormalizeImage(depth, depth_float);

		gt = cv::Mat(img.size(), CV_32SC1);

		auto start = localtimer.now();
		if (param_colorspace == 1)
			cv::cvtColor(img_float, img_float, cv::COLOR_BGR2Lab);
		else if (param_colorspace == 2)
			cv::cvtColor(img_float, img_float, cv::COLOR_BGR2HSV);
		auto elapsed = localtimer.now() - start;
		long long b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		if (param_verbosity > 0)
			printf("TIME (Color space conversion) (ms): %8.3f\n", (double)b / 1000);
		FILE *rt = fopen(RT_FILE, "a");
		fprintf(rt, "%8.3f\n", (double)b / 1000);
		fclose(rt);
		global_counter += b;

		if (param_colorspace)
			NormalizeImage(img_float, img_float);

		if (param_verbosity > 1)
			Display(img_float, "src image");
		if (param_img_need_filter)
		{
			start = localtimer.now();

			cv::medianBlur(img_float, img_filtered, param_filter_kernel_size);

			elapsed = localtimer.now() - start;
			b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			if (param_verbosity > 0)
				printf("TIME (Apply filter to image) (ms): %8.3f\n", (double)b / 1000);

			rt = fopen(RT_FILE, "a");
			fprintf(rt, "%8.3f\n", (double)b / 1000);
			fclose(rt);

			global_counter += b;
			if (param_verbosity > 1)
				Display(img_filtered, "filtered image");
		}
		else
			img_filtered = img_float;
		if (param_verbosity > 1)
			Display(depth_float, "depth image");
		if (param_depth_need_filter)
		{
			start = localtimer.now();

			cv::medianBlur(depth_float, depth_filtered, param_filter_kernel_size);

			elapsed = localtimer.now() - start;
			b = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			if (param_verbosity > 0)
				printf("TIME (Apply filter to depth) (ms): %8.3f\n", (double)b / 1000);

			rt = fopen(RT_FILE, "a");
			fprintf(rt, "%8.3f\n", (double)b / 1000);
			fclose(rt);

			global_counter += b;
			if (param_verbosity > 1)
				Display(depth_filtered, "depth median filter");
		}
		else
			depth_filtered = depth_float;

		
		//std::vector<std::list<cv::Vec2i>> partition_nomerge(n_segments);

		if (param_imread_mode & IMREAD_MODE::read_gt)
		{
			if (param_iter_test > 1)
			{
				std::vector<double> graph_params{ (double)param_pixel_vicinity, param_z_coord_weight };
				eval::TestAlgorithm(img_filtered, depth_filtered, gt, graph_params, param_min_segment_size,
					alg_params, modelling_params, clustering_params, clustering_mode,
					partition0, partition, partition_nomerge, partition_badmodel, param_iter_test);
			}
			else {}
		}
		else
		{

			RunMain(
				img_filtered,
				depth_filtered,
				param_pixel_vicinity,
				param_z_coord_weight,
				alg_params,
				partition0);

			int n_segments;
			int pixels_under_thres, seg_under_thres, num_mergers;
			int n_bad_models;

			if (clustering_mode & MODE_REMOVE)
			{
				RemoveSmallSegments(partition0, partition, param_min_segment_size, &seg_under_thres, &pixels_under_thres);
				n_segments = partition.size();

				copy_vec_to_vec(partition, partition_nomerge);
				//std::copy(partition.begin(), partition.end(), std::back_inserter(partition_nomerge));
			}
			else
			{
				copy_vec_to_vec(partition0, partition);
				//std::copy(partition0.begin(), partition0.end(), std::back_inserter(partition));
			}

			//double avg, sd;
			//PartitionStatistics(partition, &avg, &sd);
			//printf("Average segment size: %6.1f, st.dev.: %8.2f\n", avg, sd);

			if (clustering_mode & MODE_MERGE)
			{
				vector<cv::Vec3f> vnormals(partition.size());
				vector<cv::Vec4f> planes(partition.size());
				vector<double> fitting_errors(partition.size(), -1.0);

				//start = localtimer.now();

				partition_badmodel.reserve(partition.size());
				double total_fit_error = ComputePlanes(depth_filtered, partition, partition_badmodel, planes, vnormals, modelling_params, fitting_errors, &n_bad_models);

				//elapsed = localtimer.now() - start;
				//runtime += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

				// test
				FILE *f = fopen(PLANES_FILE, "a");
				fprintf(f, "==========\n");
				for (int w = 0; w < partition.size(); w++)
					fprintf(f, "Plane[%-4i]: (%13.10f, %13.10f, %13.10f, %13.10f), dot = %13.10f\n",
						w, planes[w][0], planes[w][1], planes[w][2], planes[w][3], planes[w].dot(planes[w]));
				fclose(f);

				f = fopen(RANSAC_FILE, "a");
				fprintf(f, "==========\n");
				for (int w = 0; w < partition.size(); w++)
					fprintf(f, "%12.5f\n", fitting_errors[w]);
				fprintf(f, "TOTAL: %-15.5f\n", total_fit_error);
				fclose(f);

				//start = localtimer.now();

				HAC(partition, planes, vnormals, clustering_params, &num_mergers);

				//elapsed = localtimer.now() - start;
				//runtime += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
				if (param_verbosity > -1)
					printf("TIME (TOTAL) (ms): %8.3f\n", (double)global_counter / 1000);

				rt = fopen(RT_FILE, "a");
				fprintf(rt, "%8.3f\n", (double)b / 1000);
				fclose(rt);
			}

			if (param_verbosity > 0)
			{
				printf("Found segments: %7i\n", n_segments);
				if (clustering_mode & MODE_REMOVE)
					printf("Pixels under threshold: %7i\nSegments under threshold: %7i\n",
						pixels_under_thres, seg_under_thres);
				if (clustering_mode & MODE_MERGE)
				{
					printf("# invalid planes: %5i\n", n_bad_models);
					printf("Merged from %5i to %5i\n", n_segments - n_bad_models, num_mergers);
				}
			}
			else
			{
				FILE *f = fopen(INFO_FILE, "w");
				fprintf(f, "==========\n");
				fprintf(f, "Segments initially: %7i\n", n_segments);
				if (clustering_mode & MODE_REMOVE)
					fprintf(f, "Pixels under threshold: %7i\nSegments under threshold: %7i\n",
						pixels_under_thres, seg_under_thres);
				if (clustering_mode & MODE_MERGE)
				{
					fprintf(f, "# invalid planes: %5i\n", n_bad_models);
					fprintf(f, "Merged from %5i to %5i\n", n_segments - n_bad_models, num_mergers);
				}
			}

		}

		cv::Mat colors;
		cv::Mat labels;
		if (param_verbosity > 1)
		{
			colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
			labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
			LabelPartition(partition0, labels, colors);
			PlotSegmentation(colors, 10, "segmentation - initial");

			if (clustering_mode & MODE_REMOVE)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_nomerge, labels, colors);
				PlotSegmentation(colors, 10, "segmentation - small pieces removed");
			}

			if (clustering_mode & MODE_MERGE)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition, labels, colors);
				PlotSegmentation(colors, 10, "segmentation - HAC-based");

				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_badmodel, labels, colors);
				PlotSegmentation(colors, 10, "segmentation - invalid models");

				partition.insert(partition.end(), partition_badmodel.begin(), partition_badmodel.end());
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition, labels, colors);
				PlotSegmentation(colors, 0, "segmentation - result");
			}
		}
		else
		{
			const char image_dir[] = "F:\\opticalflow\\tests\\_gt_test\\";
			char buf[100];
			colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
			labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
			LabelPartition(partition0, labels, colors);
			cv::imwrite(strcat(strcpy(buf, image_dir), "initial.png"), colors);
			if (clustering_mode & MODE_REMOVE)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_nomerge, labels, colors);
				cv::imwrite(strcat(strcpy(buf, image_dir), "no_merge.png"), colors);
			}
			if (clustering_mode & MODE_MERGE)
			{
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition, labels, colors);
				cv::imwrite(strcat(strcpy(buf, image_dir), "hac.png"), colors);
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				LabelPartition(partition_badmodel, labels, colors);
				cv::imwrite(strcat(strcpy(buf, image_dir), "invalid.png"), colors);
				colors = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
				labels = -cv::Mat::ones(img.rows, img.cols, CV_32SC1);
				partition.insert(partition.end(), partition_badmodel.begin(), partition_badmodel.end());
				LabelPartition(partition, labels, colors);
				cv::imwrite(strcat(strcpy(buf, image_dir), "result.png"), colors);
			}
		}
	}

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

//void NormalizeImage(cv::Mat & img, cv::Mat & img_float)
//{
//	double min, max;
//	if (img.channels() == 1)
//	{
//		cv::minMaxIdx(img, &min, &max);
//		if (min == max)
//			img.convertTo(img_float, CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
//		else
//			img.convertTo(img_float, CV_32FC1, 1.0 / (max - min), -min / (max - min));
//	}
//	else
//	{
//		cv::Mat channels[3];
//		cv::split(img, channels);
//		cv::minMaxIdx(channels[0], &min, &max);
//		if (min == max)
//			channels[0].convertTo(channels[0], CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
//		else
//			channels[0].convertTo(channels[0], CV_32FC1, 1.0 / (max - min), -min / (max - min));
//		cv::minMaxIdx(channels[1], &min, &max);
//		if (min == max)
//			channels[1].convertTo(channels[1], CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
//		else
//			channels[1].convertTo(channels[1], CV_32FC1, 1.0 / (max - min), -min / (max - min));
//		cv::minMaxIdx(channels[2], &min, &max);
//		if (min == max)
//			channels[2].convertTo(channels[2], CV_32FC1, min*min + max*max > 0 ? 1.0 / max : 1.0);
//		else
//			channels[2].convertTo(channels[2], CV_32FC1, 1.0 / (max - min), -min / (max - min));
//		cv::merge(channels, 3, img_float);
//	}
//}

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

void PlotSegmentation(cv::Mat &segmentation, int waittime, const char *windowname)
{
	cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
	cv::imshow(windowname, segmentation);
	cv::waitKey(waittime);
}

void PartitionStatistics(std::vector<std::list<cv::Vec2i>> &part, double *av_seg_size, double *st_dev)
{
	*av_seg_size = 0.0;
	*st_dev = 0.0;
	
	if (!part.size())
		return;

	for (int w = 0; w < part.size(); w++)
	{
		*av_seg_size += part[w].size();
		/*for (auto it = part[w].begin(); it != part[w].end(); it++)
		{}*/
	}
	*av_seg_size /= part.size();

	for (int w = 0; w < part.size(); w++)
		*st_dev += (part.size() - *av_seg_size) * (part.size() - *av_seg_size);
	*st_dev /= part.size() > 1 ? (part.size() - 1) : 1;
	*st_dev = std::sqrt(*st_dev);
}

void ReadLabelsFile(cv::Mat &gt, const char *fname)
{
	unsigned int val;
	FILE *f = fopen(fname, "rb");
	for (int u = 0; u < gt.rows; u++)
		for (int v = 0; v < gt.cols; v++)
		{
			fread(&val, sizeof(int), 1, f);
			gt.at<int>(u, v) = val;
		}
	fclose(f);
}

#else

#include <vector>
#include <list>
#include <cstring>

void ReadPCDFile(/*cv::Mat &img, */cv::Mat &labels, const char *fname)
{
	char buf[100];
	FILE *f = fopen(fname, "rb");
	// skip header
	fscanf(f, "%s %s %s %s %s %s %s %s %s\n", buf, buf, buf, buf, buf, buf, buf, buf, buf);
	fscanf(f, "%s %s\n", buf, buf);
	fscanf(f, "%s %s %s %s %s %s\n", buf, buf, buf, buf, buf, buf);
	fscanf(f, "%s %s %s %s %s %s\n", buf, buf, buf, buf, buf, buf);
	fscanf(f, "%s %s %s %s %s %s\n", buf, buf, buf, buf, buf, buf);
	fscanf(f, "%s %s %s %s %s %s\n", buf, buf, buf, buf, buf, buf);
	fscanf(f, "%s %s\n", buf, buf);
	fscanf(f, "%s %s\n", buf, buf);
	fscanf(f, "%s %s %s %s %s %s %s %s\n", buf, buf, buf, buf, buf, buf, buf, buf);
	fscanf(f, "%s %s\n", buf, buf);
	fscanf(f, "%s %s\n", buf, buf);

	//int w = 640, h = 480;
	//img = cv::Mat(h, w, CV_8UC4);
	//labels = cv::Mat(h, w, CV_32SC1);
	//float x, y, z;
	for (int i = 0; i < 480; i++)
		for (int j = 0; j < 640; j++)
		{
			fread(buf, sizeof(char), 20, f);
			
			// skip first 12 bytes
			/*x = *((float*)buf);
			y = *((float*)buf + 1);
			z = *((float*)buf + 2);*/

			/*img.at<cv::Vec4b>(i, j) = cv::Vec4b(
				*((unsigned char*)buf + 12),
				*((unsigned char*)buf + 13),
				*((unsigned char*)buf + 14),
				*((unsigned char*)buf + 15));*/
			labels.at<int>(i, j) = *((unsigned int*)buf + 4);
		}
	fclose(f);
}

//void MakeSegmentLabels(cv::Mat &labels, cv::Mat &colors)
//{
//	int a = 120, b = 255;
//	SimpleDistribution distrib(a, b);
//	map<int, cv::Vec3b> color;
//	map<int, cv::Vec3b>::iterator it;
//	int c = 0;
//
//	for (int u = 0; u < labels.rows; u++)
//		for (int v = 0; v < labels.cols; v++)
//		{
//			
//			if ((it = color.find(labels.at<int>(u, v))) == color.end())
//			{
//				cv::Vec3b h(distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()));
//				color.insert(std::make_pair(c++, h));
//				colors.at<cv::Vec3b>(u, v) = h;
//			}
//			else
//				colors.at<cv::Vec3b>(u, v) = it->second;
//		}
//}
//
//void LabelMatrix(vector<list<cv::Vec2i>> &partition, cv::Mat &labels, cv::Mat &colors)
//{
//	int a = 120, b = 255;
//	SimpleDistribution distrib(a, b);
//	map<int, cv::Vec3b> color;
//	map<int, cv::Vec3b>::iterator it;
//	int c = 0;
//	
//	for (int u = 0; u < labels.rows; u++)
//		for (int v = 0; v < labels.cols; v++)
//		{
//				
//			if ((it = color.find(labels.at<int>(u, v))) == color.end())
//			{
//				cv::Vec3b h(distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()), distrib.Get()(RNG.Get()));
//				color.insert(std::make_pair(c++, h));
//				colors.at<cv::Vec3b>(u, v) = h;
//			}
//			else
//				colors.at<cv::Vec3b>(u, v) = it->second;
//		}
//}

void WriteLabels(cv::Mat &gt, const char *fname)
{
	unsigned int val;
	FILE *f = fopen(fname, "wb");
	for (int u = 0; u < gt.rows; u++)
		for (int v = 0; v < gt.cols; v++)
			fwrite(&(val = gt.at<int>(u, v)), sizeof(int), 1, f);
	fclose(f);
}

int main(int argc, char **argv)
{
	int c = 1;
	char buf[200];
	char path[200];
	char *p, *dot;
	cv::Mat img, depth, gt(480, 640, CV_32SC1);
	//cv::Mat img_shown;
	FILE *f_image, *f_depth, *f_gt;
	f_image = fopen(argv[c++], "r");
	f_depth = fopen(argv[c++], "r");
	f_gt = fopen(argv[c++], "r");
	strcpy(path, argv[c]);
	cv::Rect roi(70, 50, 500, 430);
	cv::Size tsize(320, 240);
	while (fscanf(f_image, "%s\n", buf) != EOF)
	{
		// read rgb
		img = cv::imread(buf, cv::IMREAD_UNCHANGED);
		img = img(roi);
		cv::resize(img, img, tsize);

		p = strrchr(buf, '\\') + 1;
		//dot = strchr(p, '.');
		//strcpy(fname, p);
		cv::imwrite(strcat(path, p), img);
		strcpy(path, argv[c]);

		//read depth
		fscanf(f_depth, "%s\n", buf);
		depth = cv::imread(buf, cv::IMREAD_UNCHANGED);
		if (depth.channels() > 1)
		{
			std::vector<cv::Mat> ch;
			cv::split(depth, ch);
			depth = ch[0];
		}
		//NormalizeImage(depth, img_shown);
		//cv::imshow("depth", img_shown);
		//cv::imshow("cropped", img_shown(roi));
		//cv::waitKey(10);
		//img_shown = img_shown(roi);
		depth = depth(roi);
		cv::resize(depth, depth, tsize);
		//cv::imshow("cropped-resized", img_shown);
		//cv::waitKey(0);
		
		p = strrchr(buf, '\\') + 1;
		dot = strchr(p, '.');
		cv::imwrite(strcat(strncat(path, p, dot - p), "_depth.png"), depth);
		strcpy(path, argv[c]);

		// read pcd
		fscanf(f_gt, "%s\n", buf);
		gt = cv::Mat(480, 640, CV_32FC1);
		ReadPCDFile(gt, buf);
		gt = gt(roi);
		cv::resize(gt, gt, tsize);

		p = strrchr(buf, '\\') + 1;
		dot = strchr(p, '.');
		WriteLabels(gt, strcat(strncat(path, p, dot - p), ".labels"));
		strcpy(path, argv[c]);
	}
	fclose(f_gt);
	fclose(f_depth);
	fclose(f_image);
	/*img = cv::imread(argv[c++], cv::IMREAD_UNCHANGED);
	depth = cv::imread(argv[c++], cv::IMREAD_UNCHANGED);
	ReadPCDFile(gt, argv[c++]);*/

	return 0;
}

#endif