// ISS_detector.cpp : Defines the entry point for the console application.
//

// This code computes the ISS keypoints [1] given an input point cloud, and outputs a file (ply or txt) containing the coordinates of the keypoints
// By default, the inuput must be a ply file (mesh or point cloud). If you want to replace it by a pcd file, replace line 78 by line 79
// The second input is a scalar corresponding to the resolution of the mesh/point cloud, that is, the average distance of edges  of the mesh/the average distance between closest points of the point cloud.
// The third input is the filepath where to save the final point cloud of keypoints. 
// There are additional parameters that the user can set. These parameters have been optimized for the mean of our experiments, and so written by default inside the code. 
// If the user want to change it, he must refers from line 65 to line 70. 

// [1] : Intrinsic Shape Signatures : A shape descriptor for 3D object recognition, Zhong & all, 2009

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/iss_3d.h>
//#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
//#include <pcl/filters/voxel_grid.h>

void write_ply_file(std::string& filename, const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_ptr, const pcl::PointCloud<pcl::Normal>::Ptr& keypoints_normal_ptr)
{
	int nb_vertex = (int) keypoints_ptr->size();
	// write header
	std::ofstream output_file;
	output_file.open(filename, std::ios::out);
	output_file << "ply" << std::endl;
	output_file << "format ascii 1.0" << std::endl;
	output_file << "element vertex " << std::to_string(nb_vertex).c_str() << std::endl;
	output_file << "property float x" << std::endl;
	output_file << "property float y" << std::endl;
	output_file << "property float z" << std::endl;
	output_file << "property float nx" << std::endl;
	output_file << "property float ny" << std::endl;
	output_file << "property float nz" << std::endl;
	output_file << "end_header" << std::endl;

	// write points from kpairs_queue
	std::set<int> used_index;
	for (int i = 0; i < keypoints_ptr->size(); i++)
		output_file << (*keypoints_ptr)[i].x << " " << (*keypoints_ptr)[i].y << " " << (*keypoints_ptr)[i].z << " " << (*keypoints_normal_ptr)[i].normal_x << " " << (*keypoints_normal_ptr)[i].normal_y << " " << (*keypoints_normal_ptr)[i].normal_z << std::endl;

	output_file.close();
}


int main(int argc, const char** argv)
{

	// check the input
	if (argc < 4) {
		PCL_ERROR("you must enter first the filepath of the mesh"); 
		PCL_ERROR("you must enter secondly the resolution of the mesh (mean of edge lengths");
		PCL_ERROR("you must enter thirdly the filepath where to save keypoints (could be text file or ply file)");
		return (-1);
	}


	//  ISS3D parameters
	//
	float iss_salient_radius_; // set in line 100
	float iss_non_max_radius_; // set in line 101
	float iss_gamma_21_(0.985); // threshold value for the ratio of gamma_2 / gamma_1   (for the meaning of gamma_i, please check the original paper [1])
	float iss_gamma_32_(0.985); // threshold value for the ratio of gamma_3 / gamma_2
	float iss_min_neighbors_(5); // number minimal of neighbours that a point should have to be considered as keypoint
	int iss_threads_(4); // number of threads

	pcl::PointCloud<pcl::PointNormal>::Ptr model(new pcl::PointCloud<pcl::PointNormal>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_points(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr model_normal(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr keypoints_normal(new pcl::PointCloud<pcl::Normal>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

	// Fill in the model cloud
		// read ply file
	std::string pc_filepath = argv[1];
	std::cout << "file path : " << pc_filepath << std::endl;

	if (pcl::io::loadPLYFile<pcl::PointNormal>(pc_filepath, *model) == -1) //* load the file
	//if (pcl::io::loadPCDFile<pcl::PointXYZ>(pc_filepath, *model) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read ply file \n");
		return (-1);
	}

	std::cout << "Loaded "
		<< model->width * model->height
		<< " data points from ply file "
		<< std::endl;

	float model_resolution = std::stof(argv[2]);
	std::string output_filename = argv[3];

	// Compute model_resolution
	iss_salient_radius_ = 2.0 * model_resolution; // salient_radius is equivalent to frame_radius (the radius taken to construct the frame)
	iss_non_max_radius_ = 1.0 * model_resolution; // non_max_radius is equivalent to the minimum distance between keypoints

	//
	// Compute keypoints
	//
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
	pcl::copyPointCloud<pcl::PointNormal, pcl::PointXYZ>(*model, *model_points);
	pcl::copyPointCloud<pcl::PointNormal, pcl::Normal>(*model, *model_normal);

	iss_detector.setSearchMethod(tree);
	iss_detector.setSalientRadius(iss_salient_radius_);
	iss_detector.setNonMaxRadius(iss_non_max_radius_);
	iss_detector.setThreshold21(iss_gamma_21_);
	iss_detector.setThreshold32(iss_gamma_32_);
	iss_detector.setMinNeighbors(iss_min_neighbors_);
	iss_detector.setNumberOfThreads(iss_threads_);
	iss_detector.setInputCloud(model_points);
	iss_detector.setNormals(model_normal);
	iss_detector.compute(*keypoints);

	//get respective normals of the keypoints point cloud

	pcl::search::KdTree<pcl::PointXYZ>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZ>());
	std::vector<int> normal_indices;
	std::vector<float> k_sqr_distances;
	normal_tree->setInputCloud(model_points);
	for (int i = 0; i < keypoints->size(); i++)
	{
		std::vector<int> k_indices;
		normal_tree->nearestKSearch((*keypoints)[i], 1, k_indices, k_sqr_distances);
		normal_indices.insert(normal_indices.end(), k_indices.begin(), k_indices.end());
	}

	for (int i = 0; i < normal_indices.size(); i++)
		keypoints_normal->push_back((*model_normal)[normal_indices[i]]);


	/* // write keypoints in text file
	int nb_keypoints = keypoints->width*keypoints->height;
	std::ofstream text_file;
	text_file.open(output_filename, std::ios::out);
	text_file << nb_keypoints << std::endl;
	for (int i = 0; i < nb_keypoints; i++) {
		const pcl::PointXYZ &pt = model_keypoints->points[i];
		text_file << pt.x << " " << pt.y << " " << pt.z << std::endl;
	}
	text_file.close();*/

	// or write ply file of keypoints
	write_ply_file(output_filename, keypoints, keypoints_normal);
	//pcl::io::savePLYFileASCII<pcl::PointXYZ>(output_filename, *keypoints);


    return 0;
}

