#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <list>

#define IMAGE_WIDTH		1280
#define IMAGE_HEIGHT 	1024

using namespace cv;

struct Vec2 {
	int x, y;

	Vec2(int _x, int _y) {
		this->x = _x;
		this->y = _y;
	}
};

struct PropagateNext {
	Vec2 point = Vec2(0, 0);
	float weight;

	PropagateNext(Vec2 _point, float _weight) {
		this->point = _point;
		this->weight = _weight;
	}
};

std::vector<PropagateNext> getNeighbours(Vec2 &point) {
	std::vector<PropagateNext> result;
	int x = point.x;
	int y = point.y;

	if (x < 1280 - 1)
		result.push_back(PropagateNext(Vec2(x + 1, y), 1));
	if (y < 1024 - 1)
		result.push_back(PropagateNext(Vec2(x, y + 1), 1));
	if (x > 0)
		result.push_back(PropagateNext(Vec2(x - 1, y), 1));
	if (y > 0)
		result.push_back(PropagateNext(Vec2(x, y - 1), 1));

	return result;
}

void propagateContourPoints(Mat &image, std::vector<Vec2> &contourPoints) {
	std::list<Vec2> toVisit;

	for (auto contourPoint : contourPoints)
		toVisit.push_back(contourPoint);

	while (!toVisit.empty()) {
		Vec2 currentPoint = toVisit.back();
		toVisit.pop_back();

		unsigned char value = image.ptr<unsigned char>(currentPoint.y)[currentPoint.x];

		if (value <= 0) 
			continue;

		auto potentialNeighbours = getNeighbours(currentPoint);
		for (auto potentialNeighbour : potentialNeighbours) {
			if (image.ptr<unsigned char>(potentialNeighbour.point.y)[potentialNeighbour.point.x] == 0) {
				int newValue = std::max(0.0, value - 2.0 * potentialNeighbour.weight);
				image.ptr<unsigned char>(potentialNeighbour.point.y)[potentialNeighbour.point.x] = newValue;
				toVisit.push_front(potentialNeighbour.point);
			}
		}
	}
}

void makeDistanceRegions(Mat &image) {
	int nRows = image.rows;
	int nCols = image.cols;

	unsigned char *p;

	for (int row = 0; row < nRows; ++row) {
		p = image.ptr<unsigned char>(row);
		for (int col = 0; col < nCols; ++col) {
			int value = p[col];
			
			if (value <= 255 && value >= 240) {
				image.ptr<unsigned char>(row)[col] = 255;
			} else if (value < 240 && value >= 210) {
				image.ptr<unsigned char>(row)[col] = 225;
			} else if (value < 210 && value >= 150) {
				image.ptr<unsigned char>(row)[col] = 180;
			} else if (value < 150 && value >= 70) {
				image.ptr<unsigned char>(row)[col] = 110;
			} else if (value < 70) {
				image.ptr<unsigned char>(row)[col] = 0;
			}
		}
	}
}

std::vector<Vec2> findContourPoints(Mat &image) {
	int nRows = image.rows;
	int nCols = image.cols;

	std::vector<Vec2> result;
	unsigned char *p;

	for (int row = 0; row < nRows; ++row) {
		p = image.ptr<unsigned char>(row);
		for (int col = 0; col < nCols; ++col) {
			int value = p[col];

			if (value == 255) {
				result.push_back(Vec2(col, row));
			} else if (value > 0 && value < 255) {
				p[col] = 0;
			}
		}
	}

	return result;
}

int main(int argc, char **argv) {

	std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

	// Ordner 1 bis 15
	for (int current = 1; current <= 15; ++current) {
		std::string basePath = std::string("Data/training_1500_small_dist_field/y");
		std::string currentPath = basePath + std::to_string(current) + "/"; //+ std::string("/ground_truth/");
		auto savePath = basePath + std::to_string(current) + "/"; // + std::string("/ground_truth_distField/");
		// system((std::string("mkdir -p ") + savePath).c_str());

		std::cout << "Processing " << currentPath << std::endl; 

		char numBuff[25];
		// Bilder 0 bis 99
		for (int imageNum = 0; imageNum < 100; ++imageNum) {
			sprintf(numBuff, "%03d", imageNum);

			std::cout << imageNum << std::endl;

			Mat image;
			auto imName = "frame" + std::string(numBuff);
			// std::cout << currentPath + imName << std::endl;
			image = imread(currentPath + imName + ".png", IMREAD_GRAYSCALE);
			if (!image.data) {
				std::cout << "Error loading image" << std::endl;
				return 0;
			}

			// Rect roi(320, 28, IMAGE_WIDTH, IMAGE_HEIGHT);
			// image = image(roi);
			
			// auto points = findContourPoints(image);

			// propagateContourPoints(image, points);	
			makeDistanceRegions(image);
			
			std::string writeTo = savePath + imName + ".png";
			imwrite(writeTo, image, compression_params);
		}
	}

	return 0;
}