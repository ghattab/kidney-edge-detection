#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"

#include <stdio.h>
#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

void readCalibration(Mat& M_l, Mat& D_l, Mat& M_r, Mat& D_r, Mat& R, Mat& T, const string& calibrationFile) {
	double fx_l, fy_l;
	double cx_l, cy_l;
	vector<double> dist_l(5);
	double fx_r, fy_r;
	double cx_r, cy_r;
	vector<double> dist_r(5);
	vector<double> r(3);
	vector<double> t(3);

	string line;
	ifstream reader(calibrationFile.c_str());
	if (!reader.is_open()) {
		cout << "Cannot open " << calibrationFile << endl;
		return;
	}
	while (getline(reader, line)) {
		if (line.find("Camera-0-F: ") == 0) {
			line = line.substr(12);
			vector<string> elements = split(line, ' ');
			fx_l = atof(elements[0].c_str());
			fy_l = atof(elements[1].c_str());
		} else if (line.find("Camera-0-C: ") == 0) {
			line = line.substr(12);
			vector<string> elements = split(line, ' ');
			cx_l = atof(elements[0].c_str());
			cy_l = atof(elements[1].c_str());
		} else if (line.find("Camera-0-K: ") == 0) {
			line = line.substr(12);
			vector<string> elements = split(line, ' ');
			for (int i = 0; i < 5; i++) {
				dist_l.at(i) = atof(elements[i].c_str());
			}
		} else if (line.find("Camera-1-F: ") == 0) {
			line = line.substr(12);
			vector<string> elements = split(line, ' ');
			fx_r = atof(elements[0].c_str());
			fy_r = atof(elements[1].c_str());
		} else if (line.find("Camera-1-C: ") == 0) {
			line = line.substr(12);
			vector<string> elements = split(line, ' ');
			cx_r = atof(elements[0].c_str());
			cy_r = atof(elements[1].c_str());
		} else if (line.find("Camera-1-K: ") == 0) {
			line = line.substr(12);
			vector<string> elements = split(line, ' ');
			for (int i = 0; i < 5; i++) {
				dist_r.at(i) = atof(elements[i].c_str());
			}
		} else if (line.find("Extrinsic-Omega: ") == 0) {
			line = line.substr(17);
			vector<string> elements = split(line, ' ');
			for (int i = 0; i < 3; i++) {
				r.at(i) = atof(elements[i].c_str());
			}
		} else if (line.find("Extrinsic-T: ") == 0) {
			line = line.substr(13);
			vector<string> elements = split(line, ' ');
			for (int i = 0; i < 3; i++) {
				t.at(i) = atof(elements[i].c_str());
			}
		}
	}
	reader.close();

    M_l = (Mat_<double>(3, 3) << fx_l, 0, cx_l, 0, fy_l, cy_l, 0, 0, 1);
    D_l = Mat(dist_l, true);
    M_r = (Mat_<double>(3, 3) << fx_r, 0, cx_r, 0, fy_r, cy_r, 0, 0, 1);
    D_r = Mat(dist_r, true);
    R = Mat(r, true);
    T = Mat(t, true);
}

void crop(const Mat& im1, const Mat& im2, Mat& cropped_1, Mat& cropped_2) {
    Mat _cropped_1 = Mat(im1, Rect(320, 28, 1280, 1024));
    _cropped_1.copyTo(cropped_1);
    Mat _cropped_2 = Mat(im2, Rect(320, 28, 1280, 1024));
    _cropped_2.copyTo(cropped_2);
}

void rectify(const Mat& imL, const Mat& imR, Mat& rectified_l, Mat& rectified_r, Mat& R_l, Mat& P_l, Rect* roi_l, Rect* roi_r,
		/*calibration parameters...*/ const Mat& M_l, const Mat& D_l, const Mat& M_r, const Mat& D_r, const Mat& R, const Mat& T) {

	Size img_size = imL.size();
	double alpha = 1; 	// free scaling parameter between 0 and 1
						// alpha=0:	the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification)
						// alpha=1:	the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images
						//			(no source image pixels are lost)
						// If alpha is -1 or absent, the function performs default scaling.
    Mat R_r, P_r;
	Mat Q; // disparity-to-depth mapping matrix
    stereoRectify(M_l, D_l, M_r, D_r, img_size, R, T, R_l, R_r, P_l, P_r, Q, CALIB_ZERO_DISPARITY /*flags*/, alpha, img_size /*newImageSize*/, roi_l, roi_r);

    Mat map1_l, map2_l, map1_r, map2_r;
    initUndistortRectifyMap(M_l, D_l, R_l, P_l, img_size, CV_16SC2, map1_l, map2_l);	// computes the joint undistortion and rectification transformation and
    																					// represents the result in the form of maps for remap()
    initUndistortRectifyMap(M_r, D_r, R_r, P_r , img_size, CV_16SC2, map1_r, map2_r);

    remap(imL, rectified_l, map1_l, map2_l, INTER_LINEAR);
    remap(imR, rectified_r, map1_r, map2_r, INTER_LINEAR);
}

void improveDisp(const Mat& disp, Mat& dest) {
	equalizeHist(disp, dest);

	// squeeze
	int min = 10;
	int max = 245;
	for (int i = 0; i < dest.rows; i++) {
		for (int j = 0; j < dest.cols; j++) {
			dest.at<uchar>(i, j) = min + (((double) dest.at<uchar>(i, j)) / 255.0) * (max - min);
		}
	}


	for (int i = 0; i < dest.rows; i++) {
		for (int j = 0; j < dest.cols; j++) {
			if (disp.at<uchar>(i, j) < min || disp.at<uchar>(i, j) > max) {
				dest.at<uchar>(i, j) = disp.at<uchar>(i, j);
			}
		}
	}

	medianBlur(dest, dest, 5);
}

Mat calcDisparityMap(const Mat& imL, const Mat& imR) {
  
    Mat imL_grey, imR_grey;
    cvtColor(imL, imL_grey, COLOR_BGR2GRAY);
    cvtColor(imR, imR_grey, COLOR_BGR2GRAY);
    Size img_size = imL.size();
    int SADWindowSize = 5;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create();
  
    sgbm->setNumDisparities(((img_size.width/8) + 15) & -16);
    sgbm->setMinDisparity(0);
    sgbm->setBlockSize(SADWindowSize); //3;
    sgbm->setPreFilterCap(63);
    sgbm->setUniquenessRatio(7); //10;
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    int cn = imL.channels();
    sgbm->setP1(8*cn*SADWindowSize*SADWindowSize);
    sgbm->setP2(32*cn*SADWindowSize*SADWindowSize);
//    sgbm->fullDP = false;
  
    Mat disp, disp8, disp8_eq;
    sgbm->compute(imL_grey, imR_grey, disp);
  
    disp.convertTo(disp8, CV_8U, 255/(sgbm->getNumDisparities()*16.));
    improveDisp(disp8, disp8_eq);
  
    return disp8_eq;
}

void processFrames(const Mat& _imL, const Mat& _imR,
		/*where to store...*/ const string& dstPath,
		/*calibration parameters...*/ const Mat& M_l, const Mat& D_l, const Mat& M_r, const Mat& D_r, const Mat& R, const Mat& T) {
    Mat imL, imR;
    crop(_imL, _imR, imL, imR);
    Size img_size = imL.size();

    Mat rectified_l, rectified_r;
    Mat R_l, P_l;
    Rect roi_l, roi_r; // optional output rectangles inside the rectified images where all the pixels are valid
  
    rectify(imL, imR, rectified_l, rectified_r, R_l, P_l, &roi_l, &roi_r, M_l, D_l, M_r, D_r, R, T);
  
    Mat disp = calcDisparityMap(rectified_l, rectified_r);

    // de-rectify disparity map (so it fits the left image)
    Mat disp_distorted;
    Mat map1, map2;
    initUndistortRectifyMap(Mat(P_l, Rect(0, 0, 3, 3)), D_l, R_l.inv(), M_l, img_size, CV_16SC2, map1, map2);
    remap(disp, disp_distorted, map1, map2, INTER_LINEAR);

    // save
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);	// no compression
    imwrite(dstPath, disp_distorted, compression_params);
}

int main(int argc, char** argv) {
	string dirs[] = {	
						"Data/kidney_dataset_5/",
						"Data/kidney_dataset_6/",
						"Data/kidney_dataset_7/",
						"Data/kidney_dataset_8/",
						"Data/kidney_dataset_9/",
						"Data/kidney_dataset_10/",
						"Data/kidney_dataset_11/",
						"Data/kidney_dataset_12/",
						"Data/kidney_dataset_13/",
						"Data/kidney_dataset_14/",
						"Data/kidney_dataset_15/"
	};

	for (int i = 0; i < 10; i++) {
		string dataPath = dirs[i];
		cout << "Process frames from folder " << dataPath << endl;

		string calibrationFile = dataPath +  + "camera_calibration.txt";
		string left_frames = dataPath + "left_frames/";
		string right_frames = dataPath + "right_frames/";
		string dstPath = dataPath + "disparity/";
		string command = "mkdir -p " + dstPath;
		system(command.c_str());

		Mat M_l, D_l, M_r, D_r, R, T;
		readCalibration(M_l, D_l, M_r, D_r, R, T, calibrationFile);

		Mat imL, imR;
		char buf[25];
		for (int j = 0; j < 100; j++) {
			sprintf(buf, "%03d", j);
			string frameName = "frame" + string(buf) + ".png";

			imL = imread(left_frames + frameName, IMREAD_COLOR);
			if (!imL.data) {
				cout <<  "Could not open or find image " << left_frames + frameName << std::endl;
				return -1;
			}
			imR = imread(right_frames + frameName, IMREAD_COLOR);
			if (!imR.data) {
				cout <<  "Could not open or find image " << right_frames + frameName << std::endl;
				return -1;
			}

			string dstPath_disp = dstPath + "frame" + string(buf) + ".png";

			cout << frameName << endl;
			processFrames(imL, imR, dstPath_disp, M_l, D_l, M_r, D_r, R, T);
		}
	}
    return 0;
}
