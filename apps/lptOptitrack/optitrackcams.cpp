/*
Real-time particle tracking system:
This is the main code for the real-time particle tracking system. 
This creates the streaming pipeline of tasks including:
	1) data aquisition from camera
	2) image processing
	3) object detection in each image
	4) camera correspondence solver
	5) 3D reconstruction
	6) Temporal tracking
	7) Visualization

 copyright: Douglas Barker 2011
*/

#include "core.hpp"
#include "imageproc.hpp"
#include "datagen.hpp"
#include "dataaquisition.hpp"
#include "correspond.hpp"
#include "tracking.hpp"
#include "visualization.hpp"
//--All particle tracking functionality is loaded in namespace lpt::


using namespace std;

int main(int argc, char** argv) {
	string input = (argc > 1 ? argv[1] : "../../../data/input/");
	string output = (argc > 2 ? argv[2] : "../../../data/output/");
	//string cameras_file = input + "03_01_2021_6_cameras_test.yaml";//"10_31_2020_6_cameras_test.yaml";//"05_20_2020topview_left4_cameras.yaml";//"02_20_2020_8_cameras.yaml";//"01_26_2020_8_cameras.yaml";//"01_03_2020_8_cameras.yaml";//"12_31_2019_8_cameras.yaml";//"11_02_2019_8_cameras.yaml";//"10_31_2019_8cameras.yaml";//"11_02_25_2019topview_outside_4_cameras .yaml";//"10_28_2019_8cameras.yaml";//"09_28_2019_8cameras.yaml";//"09_25_2019topview_right4_cameras.yaml";//"09_28_2019_8cameras.yaml";//"09_25_2019topview_left4_cameras.yaml";//"09_26_2019_8cameras.yaml";//"09_25_2019topview_left4_cameras.yaml";//"09_25_2019_8cameras.yaml";////"09_17_2019topview_cameras.yaml";//"08_31_2019_8cameras.yaml";//"8cameras_08_28_2019.yaml";//"left_4_cameras.yaml";//"07_19_2019cameras.yaml";//"8_cameras.yaml";//"07_08_2019cameras.yaml";//"4_cameras_04_16_2019.yaml";//"04_15_2019.yaml";//"4_11_6.yaml";//"4_11_3.yaml";//"8_cameras.yaml"; //"../../../data/pivstd/4cams/cameras.yaml"; //4_cameras.yaml
	//string camera_pairs_file = input + "03_01_2021_6_pairs_test.yaml"; //"10_31_2020_6_pairs_test.yaml";//"05_20_2020topview_left4_pairs.yaml";//"02_20_2020_8_pairs.yaml";//"01_03_2020_8_pairs.yaml";//"12_31_2019_8_pairs.yaml";//"11_02_2019_8_pairs.yaml";//"10_31_2019_8pairs.yaml";//"11_02_25_2019topview_outside_4_pairs.yaml";//"09_28_2019_8pairs.yaml";//"09_25_2019topview_right4_pairs.yaml";//"09_28_2019_8pairs.yaml";//"09_25_2019topview_left4_pairs.yaml";//"09_26_2019_8pairs.yaml";//"09_25_2019topview_left4_pairs.yaml";//////"09_17_2019topview_pairs.yaml";//"08_31_2019_8pairs.yaml";//"8pairs_08_28_2019.yaml";//"right_4_pairs.yaml";//"07_19_2019pairs.yaml";//"8_pairs.yaml";////"07_08_2019pairs.yaml";//"4_cameras_pairs_04_16_2019.yaml";//"4_pairs_11_12.yaml";//"4_pairs_11_3.yaml";//"8_pairs.yaml"; //"../../../data/pivstd/4cams/camera_pairs.yaml"; ////4_pairs.yaml
	//

	string cameras_file = input + "cameras.yaml";
	string camera_pairs_file = input + "pairs.yaml";

	//string cameras_file = input + "cameras_calibrated.yaml";
	//string camera_pairs_file = input + "pairs_calibrated.yaml";


	lpt::StreamingPipeline pipeline;
	pipeline.setQueueCapacity(1000);

	auto processor = lpt::ImageProcessor::create();
    lpt::ImageProcess::Ptr blur = lpt::GaussianBlur::create(3);    //# TODO uncomment gaussian blur later if not using svm
    lpt::ImageProcess::Ptr thresh = lpt::Threshold::create(50);  // uncomment when using FC detector
	//lpt::ImageProcess::Ptr resize = lpt::Resize::create(cv::Size(), 2, 2, cv::INTER_LINEAR);
	processor->addProcess( blur );
	processor->addProcess( thresh );
	//processor->addProcess(resize);
	
	auto detector = lpt::FindContoursDetector::create();
	//auto detector = lpt::GoodFeaturesToTrackDetector::create();

	auto camera_system = lpt::Optitrack::create();

	auto matcher = lpt::PointMatcher::create();
	matcher->params.match_threshold = 2.0;
	// matcher->params.match_thresh_level = 20;

	auto matcher_cuda =  lpt::PointMatcherCUDA::create();
	// matcher_cuda->params.match_threshold = 5.0; //pixels
	// matcher_cuda->params.match_thresh_level = 20;

	auto tracker = lpt::Tracker::create();
    tracker->setCostCalculator(lpt::CostMinimumAcceleration::create());
	tracker->params.min_radius = 4.0; //mm
	tracker->params.min_radius_level = 4;
	tracker->params.max_radius = 25.0; //mm
	tracker->params.max_radius_level = 25;
    tracker->params.KF_sigma_a = 2.75E-4;
    tracker->params.KF_sigma_z = 1E-2;
	//
    bool KalmanFilter = false;

	auto visualizer = lpt::Visualizer::create();
	double grid_size[3] = {310.0, 310.0, 610.0};
	double grid_width = 10.0;
	int cell_counts[3] = {31, 31, 61};
	visualizer->getVolumeGrid()->setGridOrigin(-155.00, -155.00, -355.0);

	visualizer->getVolumeGrid()->setGridCellCounts(cell_counts[0], cell_counts[1], cell_counts[2]);
	visualizer->getVolumeGrid()->setGridDimensions(grid_size[0], grid_size[1], grid_size[2]);
	
	pipeline.setInputDataPath(input);
	pipeline.setOutputDataPath(output);
	pipeline.load_Rotation_Matrix();
    pipeline.setKalmanFilter(KalmanFilter);
	pipeline.attachCameraSystem(camera_system);
	pipeline.attachImageProcessor(processor);
	pipeline.attachDetector(detector);
	pipeline.attachMatcher(matcher_cuda);
	pipeline.attachTracker(tracker);
	pipeline.attachVisualizer(visualizer);
	
	bool on = pipeline.initialize();

	if (on){ 
		camera_system->loadCameraParams(cameras_file);
		camera_system->loadCameraPairParams(camera_pairs_file);
		pipeline.run();
	} else {
		cout << "System could not initialize: Shutting down" << endl;
	}
	cout.clear();
	cout << "Finished" << endl;
	return 0;
}

