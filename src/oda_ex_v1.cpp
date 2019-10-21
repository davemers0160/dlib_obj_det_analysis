#define _CRT_SECURE_NO_WARNINGS

// #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
// #include <windows.h>
// #endif

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <utility>
#include <stdexcept>
#include <tuple>

// Custom includes
#include "obj_detector.h"
#include "get_platform.h"
//#include "file_parser.h"
#include "get_current_time.h"
#include "gorgon_capture.h"
#include "num2string.h"
#include "overlay_bounding_box.h"
//#include "array_image_operations.h"

// Net Version
#include "yj_net_v4.h"
#include "load_data.h"
#include "eval_net_performance.h"
//#include "enhanced_array_cropper.h"
//#include "random_channel_swap.h"
//#include "enhanced_channel_swap.h"


// dlib includes
//#include "random_array_cropper.h"
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/rand.h>

// new copy and set learning rate includes
//#include "copy_dlib_net.h"
//#include "dlib_set_learning_rates.h"


// -------------------------------GLOBALS--------------------------------------

std::string platform;

//this will store the standard RGB images
std::vector<dlib::matrix<dlib::rgb_pixel>> train_images, test_images;

// this will store the ground truth data for the bounding box labels
std::vector<std::vector<dlib::mmod_rect>> train_labels, test_labels;

std::string version;
//std::string net_name = "yj_net_";
//std::string net_sync_name = "yj_sync_";
std::string logfileName = "oda_log_";


// ----------------------------------------------------------------------------
void get_platform_control(void)
{
	get_platform(platform);

	if (platform == "")
	{
		std::cout << "No Platform could be identified... defaulting to Windows." << std::endl;
		platform = "Win";
	}

	//version = version + platform;
	// net_sync_name = version + "_sync";
	logfileName = logfileName + version;
	// net_name = version +  "_final_net.dat";
}

// ----------------------------------------------------------------------------------------

void print_usage(void)
{
	std::cout << "Enter the following as arguments into the program:" << std::endl;
	std::cout << "<image file name> " << std::endl;
	std::cout << endl;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{

	uint64_t idx = 0, jdx = 0;
    uint8_t HPC = 0;
	std::string sdate, stime;

    // data IO variables
	const std::string os_file_sep = "/";
	std::string program_root;
    //std::string &data_file;
    std::string network_weights_file;
	std::string image_save_location;
	std::string results_save_location;
	std::string test_inputfile;
	std::string test_data_directory;
	std::vector<std::vector<std::string>> test_file;
	std::vector<std::string> te_image_files;
	std::ofstream DataLogStream;

    // timing variables
	typedef std::chrono::duration<double> d_sec;
	auto start_time = chrono::system_clock::now();
	auto stop_time = chrono::system_clock::now();
	auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);



    std::pair<uint32_t, uint32_t> target_size(45, 100);
	//uint32_t min_target_size = 45;  // 20 min_object_length_short_dimension
	//uint32_t max_target_size = 100;  // 70 min_object_length_long_dimension
	//std::vector<int32_t> gpu;
    //uint64_t one_step_calls = 0;
    //uint64_t epoch = 0;
    //uint64_t index = 0;   

    //create window to display images
    dlib::image_window win;
    dlib::rgb_pixel color;
    dlib::matrix<dlib::rgb_pixel> tmp_img;

    // set the learning rate multipliers: 0 means freeze the layers; r1 = learning rate multiplier, r2 = learning rate bias multiplier
    //double r1 = 1.0, r2 = 1.0;

    dlib::rand rnd;
    rnd = dlib::rand(time(NULL));
    
    // ----------------------------------------------------------------------------------------
   
	if (argc == 1)
	{
		print_usage();
		std::cin.ignore();
		return 0;
	}

	std::string parse_filename = argv[1];

	// parse through the supplied csv file
	parse_input_file(parse_filename, test_inputfile, network_weights_file, version, results_save_location);

	// check the platform
	get_platform_control();

    // check for HPC <- set the environment variable PLATFORM to HPC
    if(platform.compare(0,3,"HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

	// setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
	program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
	//sync_save_location = program_root + "nets/";
    //results_save_location = program_root + "results/";
	//image_save_location = program_root + "result_images/";

#else
	if (HPC == 1)
	{
		//HPC version
		program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
	}
	else
	{
		// Ubuntu
        if(platform.compare(0,8,"MainGear") == 0)
		{
            program_root = "/home/owner/Projects/machineLearningResearch/";
        }
        else
        {
            if (platform.compare(0,7,"SL02319") == 0)
            {
                // fill in the location of where the root program is running
                program_root = "/media/daleas/DATA/Ashley_ML/machineLearningResearch/";
            }
            else
            {
                // fill in the location of where the root program is running
                program_root = "/mnt/data/machineLearningResearch/";
            }

        }
	}

	//sync_save_location = program_root + "nets/";
	//results_save_location = program_root + "results/";
	//image_save_location = program_root + "result_images/";

#endif

	std::cout << "Reading Inputs... " << std::endl;
	std::cout << "Platform:              " << platform << std::endl;
    //std::cout << "GPU:                   { ";
    //for (idx = 0; idx < gpu.size(); ++idx)
    //    std::cout << gpu[idx] << " ";
    //std::cout << "}" << std::endl;
	std::cout << "program_root:          " << program_root << std::endl;
	//std::cout << "sync_save_location:    " << sync_save_location << std::endl;
	std::cout << "results_save_location: " << results_save_location << std::endl;
    //std::cout << "image_save_location:   " << image_save_location << std::endl;


	try {

		get_current_time(sdate, stime);
		logfileName = logfileName + sdate + "_" + stime + ".txt";
        //cropper_stats_file = output_save_location + "cr_stats_" + version + "_" + sdate + "_" + stime + ".txt";

		std::cout << "Log File:              " << (results_save_location + logfileName) << std::endl << std::endl;
		DataLogStream.open((results_save_location + logfileName), ios::out | ios::app);

		// Add the date and time to the start of the log file
		DataLogStream << "------------------------------------------------------------------" << std::endl;
		DataLogStream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
		DataLogStream << "Platform: " << platform << std::endl;
        //DataLogStream << "GPU: { ";
        //for (idx = 0; idx < gpu.size(); ++idx)
        //    DataLogStream << gpu[idx] << " ";
        //DataLogStream << "}" << std::endl;
		DataLogStream << "------------------------------------------------------------------" << std::endl;

		///////////////////////////////////////////////////////////////////////////////
		// Step 1: Read in the test images
		///////////////////////////////////////////////////////////////////////////////

        parse_group_csv_file(test_inputfile, '{', '}', test_file);
        if (test_inputfile.size() == 0)
        {
            throw std::exception("Test file is empty");
        }

        // the data directory should be the first entry in the input file
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        test_data_directory = test_file[0][0];
#else
        if (HPC == 1)
        {
            test_data_directory = test_file[0][2];
        }
        else if (platform.compare(0,7,"SL02319") == 0)
        {
            test_data_directory = test_file[0][2];
        }
        else
        {
            test_data_directory = test_file[0][1];
        }
#endif

		test_file.erase(test_file.begin());
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
		std::cout << "data_directory:        " << test_data_directory << std::endl;
        std::cout << "test input file:       " << test_inputfile << std::endl;
		std::cout << "Test image sets to parse: " << test_file.size() << std::endl;

        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << test_inputfile << std::endl;
        DataLogStream << "Test image sets to parse: " << test_file.size() << std::endl;


        std::cout << "Loading test images... ";

		// load in the images and labels
        start_time = chrono::system_clock::now();
        load_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        DataLogStream << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;


        // ------------------------------------------------------------------------------------

        // for debugging to view the images
        //for (idx = 0; idx < test_images.size(); ++idx)
        //{

        //    win.clear_overlay();
        //    win.set_image(test_images[idx]);

        //    for (jdx = 0; jdx < test_labels[idx].size(); ++jdx)
        //    {
        //        color = test_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
        //        win.add_overlay(test_labels[idx][jdx].rect, color);
        //    }

        //    win.set_title(("Training Image: " + num2str(idx+1,"%05u")));

        //    std::cin.ignore();
        //    //dlib::sleep(800);
        //}

        ///////////////////////////////////////////////////////////////////////////////
        // Step 2: Setup the network
        ///////////////////////////////////////////////////////////////////////////////

        // this sets th GPUs to use algorithms that are smaller in memory but may take a little longer to execute
        dlib::set_dnn_prefer_smallest_algorithms();

        // load the network from the saved file
        aobj_net_type test_net;

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl; 
        std::cout << "Loading network: " << (network_weights_file) << std::endl;
        dlib::deserialize(network_weights_file) >> test_net;

//		// Now we are ready to create our network and trainer.
//#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
//        
//
//		// load in the convolutional filter numbers from the input file
//        //config_net(net, options, filter_num);
//#else
//        // check for the gcc version
//        #if defined(__GNUC__) && (__GNUC__ > 5)
//            obj_net_type net;
//            //config_net(net, options, filter_num);
//        #else
//            obj_net_type net;
//            //config_net(net, options, filter_num);
//        #endif
//#endif

        // show the network to verify that it looks correct
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        //std::cout << "Net Name: " << net_name << std::endl;
        std::cout << test_net << std::endl;

        //DataLogStream << "Net Name: " << net_name << std::endl;
        DataLogStream << test_net << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        // get the details about the loss layer -> the number and names of the classes
        dlib::mmod_options options = dlib::layer<0>(test_net).loss_details().get_options();

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::set<std::string> tmp_names;
        for (auto& w : options.detector_windows)
        {
            std::cout << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            DataLogStream << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            tmp_names.insert(w.label);
        }

        std::vector<std::string> class_names(tmp_names.begin(), tmp_names.end());
        uint32_t num_classes = class_names.size();
        std::vector<label_stats> test_label_stats(num_classes, label_stats(0, 0));

        std::vector<dlib::rgb_pixel> class_color;
        for (idx = 0; idx < num_classes; ++idx)
        {
            class_color.push_back(dlib::rgb_pixel(rnd.get_random_8bit_number(), rnd.get_random_8bit_number(), rnd.get_random_8bit_number()));
        }

//-----------------------------------------------------------------------------
// TRAINING START
//-----------------------------------------------------------------------------

        // this matrix will contain the results of the training and testing
		dlib::matrix<double, 1, 6> test_results = dlib::zeros_matrix<double>(1, 6);


//-----------------------------------------------------------------------------
//          EVALUATE THE FINAL NETWORK PERFORMANCE
//-----------------------------------------------------------------------------

        std::cout << std::endl << "Analyzing Test Results..." << std::endl;

		for (idx = 0; idx < test_images.size(); ++idx)
        {
            //test_label.clear();
            //load_single_set(test_data_directory, test_file[idx], test_image, test_label);

            //merge_channels(test_image, tmp_img);
			//std::cout << te_image_files[idx].first;


            std::vector<dlib::mmod_rect> dnn_labels;
            std::vector<label_stats> ls(num_classes, label_stats(0, 0));

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            dlib::matrix<double, 1, 6> tr = eval_net_performance(test_net, test_images[idx], test_labels[idx], dnn_labels, target_size.first, fda_test_box_overlap(0.4, 1.0), class_names, ls);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            const auto& layer_output_1 = dlib::layer<1>(test_net).get_output();
            const float* data_1 = layer_output_1.host();

            //const auto& layer_output_e = dlib::layer<aobj_net_type::num_layers - 1>(test_net);
            auto op = dlib::layer<20>(test_net).get_pyramid_outer_padding();
            auto pd = dlib::layer<20>(test_net).get_pyramid_padding();

            std::vector<dlib::rectangle> rects;
            dlib::matrix<dlib::rgb_pixel> tiled_img;
            // Get the type of pyramid the CNN used
            using pyramid_type = std::remove_reference<decltype(dlib::input_layer(test_net))>::type::pyramid_type;
            // And tell create_tiled_pyramid to create the pyramid using that pyramid type.
            dlib::create_tiled_pyramid<pyramid_type>(test_images[idx], tiled_img, rects,
                dlib::input_layer(test_net).get_pyramid_padding(),
                dlib::input_layer(test_net).get_pyramid_outer_padding());

            win.clear_overlay();
            win.set_image(tiled_img);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            std::cout << "Image Size (h x w): " << test_images[idx].nr() << "x" << test_images[idx].nc() << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;

            DataLogStream << "------------------------------------------------------------------" << std::endl;
            DataLogStream << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            DataLogStream << "Image Size (h x w): " << test_images[idx].nr() << "x" << test_images[idx].nc() << std::endl;
            DataLogStream << "Classification Time (s): " << elapsed_time.count() << std::endl;

            for (jdx = 0; jdx < num_classes; ++jdx)
            {
                test_label_stats[jdx].count += ls[jdx].count;
                test_label_stats[jdx].match_count += ls[jdx].match_count; 
                
                double acc = (ls[jdx].count == 0) ? 0.0 : ls[jdx].match_count / (double)ls[jdx].count;
                std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << ls[jdx].match_count << ", " << ls[jdx].count << std::endl;
                DataLogStream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << ls[jdx].match_count << ", " << ls[jdx].count  << std::endl;
            }

            std::cout << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;
            DataLogStream << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            //dnn_test_labels.push_back(dnn_labels);

			win.clear_overlay();
			//win.set_image(test_images[idx]);

            // copy the image into tmp_img so that the original data is not modified
            dlib::assign_image(tmp_img, test_images[idx]);

            //overlay the dnn detections on the image
            for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
            {
                //win.add_overlay(dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0));
                //draw_rectangle(tmp_img, dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0), 2);

                auto& class_index = std::find(class_names.begin(), class_names.end(), dnn_labels[jdx].label);
                overlay_bounding_box(tmp_img, dnn_labels[jdx], class_color[std::distance(class_names.begin(), class_index)]);

				DataLogStream << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
				std::cout << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
            }

            win.set_image(tmp_img);
            
            //// overlay the ground truth boxes on the image
            //for (jdx = 0; jdx < test_labels[idx].size(); ++jdx)
            //{
            //    win.add_overlay(test_labels[idx][jdx].rect, dlib::rgb_pixel(0, 255, 0));
            //    draw_rectangle(tmp_img, test_labels[idx][jdx].rect, dlib::rgb_pixel(0, 255, 0), 2);
            //}
            
            //save results in image form
            std::string image_save_name = image_save_location + "test_img_" + version + num2str(idx, "_%05d.png");
            save_png(tmp_img, image_save_name);

            test_results += tr;
            dlib::sleep(50);
            std::cin.ignore();

		}

/*
        // output the test results
        std::cout << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        for (jdx = 0; jdx < num_classes; ++jdx)
        {
            double acc = (test_label_stats[jdx].count == 0) ? 0.0 : test_label_stats[jdx].match_count / (double)test_label_stats[jdx].count;
            std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << std::endl;
            //DataLogStream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << std::endl;
        }
        std::cout << "Testing Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0) / (double)test_file.size() << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;


        // save the results to the log file
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Testing Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0) / (double)test_file.size() << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        DataLogStream << "class_name, detction_accuracy, correct_detects, groundtruth" << std::endl;
        for (jdx = 0; jdx < num_classes; ++jdx)
        {
            double acc = (test_label_stats[jdx].count == 0) ? 0.0 : test_label_stats[jdx].match_count / (double)test_label_stats[jdx].count;
            DataLogStream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << std::endl;
        }
        DataLogStream << "------------------------------------------------------------------" << std::endl;

*/
        std::cout << "End of Program." << std::endl;
        DataLogStream.close();
        std::cin.ignore();
        
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;

		DataLogStream << std::endl;
		DataLogStream << "------------------------------------------------------------------" << std::endl;
		DataLogStream << e.what() << std::endl;
		DataLogStream << "------------------------------------------------------------------" << std::endl;
		DataLogStream.close();

		std::cout << "Press Enter to close..." << std::endl;
		std::cin.ignore();
	}

	return 0;

}
