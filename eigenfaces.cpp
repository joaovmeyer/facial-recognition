#include <vector>

// all from ML
#include "../graph.h"
#include "../matrix.h"
#include "../dataset.h"
#include "../PCA.h"
#include "../kd-tree.h"

using namespace std;


int toGrayscale(int r, int g, int b) {
	return static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
}

Vec getImageGrayscale(const string& src) {
	olc::Sprite sprite;

	if (!sprite.LoadFromFile(src)) {
		cout << "unable to load image.\n";
		return Vec();
	}

	int w = sprite.width;
	int h = sprite.height;

	Vec img = Vec::zeros(h * w);

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			olc::Pixel pixel = sprite.GetPixel(x, y);

			img[y * w + x] = toGrayscale(pixel.r, pixel.g, pixel.b);
		}
	}

	return img;
}



void drawImage(olc::Sprite* sprite, double x0, double y0, double x1, double y1, Graph& graph, const Vec& image, size_t imgWidth) {
	for (size_t i = 0; i < image.size; ++i) {
		int x = i % imgWidth;
		int y = static_cast<double>(i) / static_cast<double>(imgWidth);

		sprite->SetPixel(x, y, olc::Pixel(image[i], image[i], image[i]));
	}

	// display the compressed image
	graph.addImage(sprite, x0, y0, x1, y1);
}






int main() {

	Graph graph;
	Dataset faces;

	// width and height of the images
	int width = 70;
	int height = 80;

	// generating the dataset
	for (int i = 0; i < 40; ++i) {

		Vec classification = Vec::zeros(40);
		classification[i] = 1;

		for (int j = 1; j <= 10; ++j) {
			string src = "datasets/olivetti faces/" + std::to_string(10 * i + j) + "_" + std::to_string(i + 1) + ".jpg";

			faces.add(DataPoint(getImageGrayscale(src), classification));
		}
	}

	// split the dataset into training (70%) and testing (30%)
	vector<Dataset> parts = Dataset::split(faces, { 70, 30 }, true);
	Dataset training = parts[0], testing = parts[1];


	PCA model;
	model.fit(training, 20);


	// transform every image on the training and testing datasets to the eigenfaces base
	Dataset trainingTransformed = model.transform(training);
	Dataset testingTransformed = model.transform(testing);

	// I'll use a kd-tree to perform a KNN search because I'm lazy, probably not worth it
	KDTree tree = KDTree::build(trainingTransformed);

	// make array of sprites to display the predictions
	olc::Sprite predictions[testing.size * 3 + 10 * 3];
	for (size_t i = 0; i < testing.size * 3 + 10 * 3; ++i) {
		predictions[i] = olc::Sprite(width, height);
	}

	size_t corrects = 0;
	for (size_t i = 0; i < testing.size; ++i) {

		DataPoint pred = *tree.getKNN(testingTransformed[i], 1)[0];

		size_t index = i - corrects;
		double posY = index;
		double posX = 0.5;

		if (pred.y == testingTransformed[i].y) {
			index = testing.size + corrects;
			posY = corrects++;
			posX = 4.5;

			// only display first 10 correct predictions
			if (corrects > 10) {
				continue;
			}
		}

		size_t j;
		for (j = 0; j < training.size; ++j) {
			if (trainingTransformed[j].x == pred.x) {
				break;
			}
		}

		// draw original image
		drawImage(&predictions[index * 3 + 0], posX, -1.5 - 1.1 * posY, posX + 1, -0.5 - 1.1 * posY, graph, testing[i].x, width);

		// draw reconstructed image
		drawImage(&predictions[index * 3 + 1], posX + 1.1, -1.5 - 1.1 * posY, posX + 2.1, -0.5 - 1.1 * posY, graph, model.toOriginalSpace(testingTransformed[i]).x, width);

		// draw predicted image
		drawImage(&predictions[index * 3 + 2], posX + 2.2, -1.5 - 1.1 * posY, posX + 3.2, -0.5 - 1.1 * posY, graph, training[j].x, width);
	}
	cout << "Got " << corrects << " correct answers out of " << testing.size << ". This is " <<  100 * static_cast<double>(corrects) / static_cast<double>(testing.size) << "% accuracy.\n";





	// display the mean face
	Vec meanFace = model.mean;
	olc::Sprite meanFaceSprite(width, height);
	for (size_t i = 0; i < faces.dimX; ++i) {

		int x = i % width;
		int y = static_cast<double>(i) / static_cast<double>(width);

		meanFaceSprite.SetPixel(x, y, olc::Pixel(meanFace[i], meanFace[i], meanFace[i]));
	}
	graph.addImage(&meanFaceSprite, 1, 4, 2, 5);


	// display the eigenfaces used
	olc::Sprite eigenfaces[20];
	for (size_t n = 0; n < 20; ++n) {

		// put every eigenface in the [0, 255] range
		Vec eigenface = Vec::zeros(faces.dimX);
		double min = model.base[0][n];
		double max = model.base[0][n];
		for (size_t i = 0; i < faces.dimX; ++i) {
			eigenface[i] = model.base[i][n];

			min = std::min(eigenface[i], min);
			max = std::max(eigenface[i], max);
		}

		for (size_t i = 0; i < faces.dimX; ++i) {
			eigenface[i] = 255 * (eigenface[i] - min) / (max - min);
		}

		// make the sprite
		eigenfaces[n] = olc::Sprite(width, height);
		for (size_t i = 0; i < faces.dimX; ++i) {

			int x = i % width;
			int y = static_cast<double>(i) / static_cast<double>(width);

			eigenfaces[n].SetPixel(x, y, olc::Pixel(eigenface[i], eigenface[i], eigenface[i]));
		}

		graph.addImage(&eigenfaces[n], 2.1 + (n % 5) * 1.1, 4 - 1.1 * (n % 4), 3.1 + (n % 5) * 1.1, 5 - 1.1 * (n % 4));
	}

	graph.waitFinish();

	return 0;
}
