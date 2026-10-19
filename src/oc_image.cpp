/*
 * This file is part of OpenCorr, an open source C++ library for
 * study and development of 2D, 3D/stereo and volumetric
 * digital image correlation.
 *
 * Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one from http://mozilla.org/MPL/2.0/.
 *
 * More information about OpenCorr can be found at https://www.opencorr.org/
 */
// Modified for OpenCorr-GPU; see NOTICE.md for the list of changes.

#include <fstream>
#include <limits>
#include <omp.h>

#include "oc_image.h"

namespace opencorr
{
	const char* imageInputDepthModeName(ImageInputDepthMode depth)
	{
		switch (depth)
		{
		case ImageInputDepthMode::Compat8:
			return "8";
		case ImageInputDepthMode::Depth12:
			return "12";
		case ImageInputDepthMode::Depth16:
			return "16";
		case ImageInputDepthMode::Auto:
			return "auto";
		}
		return "unknown";
	}

	const char* imageInputAlignmentName(ImageInputAlignment alignment)
	{
		switch (alignment)
		{
		case ImageInputAlignment::Right:
			return "right";
		case ImageInputAlignment::Left:
			return "left";
		case ImageInputAlignment::Auto:
			return "auto";
		}
		return "unknown";
	}

	const char* imageInputNormalizeName(ImageInputNormalize normalize)
	{
		switch (normalize)
		{
		case ImageInputNormalize::None:
			return "none";
		case ImageInputNormalize::To8:
			return "to-8";
		case ImageInputNormalize::To16:
			return "to-16";
		case ImageInputNormalize::Unit:
			return "unit";
		}
		return "unknown";
	}

	std::string imageCvDepthName(int depth)
	{
		switch (depth)
		{
		case CV_8U:
			return "CV_8U";
		case CV_8S:
			return "CV_8S";
		case CV_16U:
			return "CV_16U";
		case CV_16S:
			return "CV_16S";
		case CV_32S:
			return "CV_32S";
		case CV_32F:
			return "CV_32F";
		case CV_64F:
			return "CV_64F";
		default:
			return "CV_UNKNOWN";
		}
	}

	namespace
	{
		ImageInputDepthMode inferAutoDepth(int source_depth, double source_max)
		{
			if (source_depth == CV_8U)
			{
				return ImageInputDepthMode::Compat8;
			}
			if (source_depth == CV_16U && source_max <= 4095.0)
			{
				return ImageInputDepthMode::Depth12;
			}
			return ImageInputDepthMode::Depth16;
		}

		double logicalMaxForDepth(ImageInputDepthMode depth)
		{
			switch (depth)
			{
			case ImageInputDepthMode::Compat8:
				return 255.0;
			case ImageInputDepthMode::Depth12:
				return 4095.0;
			case ImageInputDepthMode::Depth16:
				return 65535.0;
			case ImageInputDepthMode::Auto:
				break;
			}
			return 255.0;
		}
	}

	cv::Mat loadImage2DMat(std::string file_path, const ImageLoadOptions& options,
		ImageLoadInfo* info)
	{
		ImageLoadInfo local_info;
		local_info.requested_depth = options.depth;
		local_info.requested_alignment = options.alignment;
		local_info.normalize = options.normalize;
		local_info.used_anydepth = options.depth != ImageInputDepthMode::Compat8;
		local_info.imread_flags = local_info.used_anydepth
			? (cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE)
			: cv::IMREAD_GRAYSCALE;

		cv::Mat loaded = cv::imread(file_path, local_info.imread_flags);
		if (!loaded.data)
		{
			throw std::string("Fail to load file: " + file_path);
		}

		local_info.source_depth = loaded.depth();
		local_info.source_channels = loaded.channels();
		cv::minMaxLoc(loaded, &local_info.source_min, &local_info.source_max);

		local_info.effective_depth = options.depth == ImageInputDepthMode::Auto
			? inferAutoDepth(local_info.source_depth, local_info.source_max)
			: options.depth;
		local_info.effective_alignment = ImageInputAlignment::Auto;

		cv::Mat output;
		if (options.depth == ImageInputDepthMode::Compat8
			&& options.normalize == ImageInputNormalize::None)
		{
			output = loaded;
		}
		else
		{
			loaded.convertTo(output, CV_32F);

			double scale = 1.0;
			if (local_info.effective_depth == ImageInputDepthMode::Depth12)
			{
				ImageInputAlignment alignment = options.alignment;
				if (alignment == ImageInputAlignment::Auto)
				{
					alignment = local_info.source_max > 4095.0
						? ImageInputAlignment::Left
						: ImageInputAlignment::Right;
				}
				local_info.effective_alignment = alignment;

				if (alignment == ImageInputAlignment::Right)
				{
					if (local_info.source_max > 4095.0)
					{
						throw std::string("12-bit right-aligned input has values above 4095: " + file_path);
					}
				}
				else if (alignment == ImageInputAlignment::Left)
				{
					scale *= 1.0 / 16.0;
				}
			}

			double logical_max = logicalMaxForDepth(local_info.effective_depth);
			switch (options.normalize)
			{
			case ImageInputNormalize::None:
				break;
			case ImageInputNormalize::To8:
				scale *= 255.0 / logical_max;
				break;
			case ImageInputNormalize::To16:
				scale *= 65535.0 / logical_max;
				break;
			case ImageInputNormalize::Unit:
				scale *= 1.0 / logical_max;
				break;
			}

			local_info.output_scale = scale;
			if (scale != 1.0)
			{
				output *= scale;
			}
		}

		local_info.output_depth = output.depth();
		cv::minMaxLoc(output, &local_info.output_min, &local_info.output_max);
		if (info != nullptr)
		{
			*info = local_info;
		}
		return output;
	}

	//2D image
	Image2D::Image2D(int width, int height)
	{
		cv_mat = cv::Mat::zeros(height, width, CV_8UC1);
		eg_mat = Eigen::MatrixXf::Zero(height, width);
		this->width = width;
		this->height = height;
		size = height * width;
	}

	Image2D::Image2D(std::string file_path) : height(0), width(0)
	{
		load(file_path);
	}

	void Image2D::load(std::string file_path)
	{
		cv_mat = cv::imread(file_path, cv::IMREAD_GRAYSCALE);

		if (!cv_mat.data)
		{
			throw std::string("Fail to load file: " + file_path);
		}

		this->file_path = file_path;
		load_info = ImageLoadInfo();
		load_info.source_depth = cv_mat.depth();
		load_info.source_channels = cv_mat.channels();
		load_info.output_depth = cv_mat.depth();
		cv::minMaxLoc(cv_mat, &load_info.source_min, &load_info.source_max);
		load_info.output_min = load_info.source_min;
		load_info.output_max = load_info.source_max;

		if (width != cv_mat.cols || height != cv_mat.rows)
		{
			width = cv_mat.cols;
			height = cv_mat.rows;
			size = height * width;
			eg_mat.resize(height, width);
		}

		cv::cv2eigen(cv_mat, eg_mat);
	}

	void Image2D::loadWithOptions(std::string file_path, const ImageLoadOptions& options,
		ImageLoadInfo* info)
	{
		ImageLoadInfo local_info;
		cv_mat = loadImage2DMat(file_path, options, &local_info);

		this->file_path = file_path;
		load_info = local_info;

		if (width != cv_mat.cols || height != cv_mat.rows)
		{
			width = cv_mat.cols;
			height = cv_mat.rows;
			size = height * width;
			eg_mat.resize(height, width);
		}

		cv::cv2eigen(cv_mat, eg_mat);
		if (info != nullptr)
		{
			*info = local_info;
		}
	}



	//3D image
	Image3D::Image3D(int dim_x, int dim_y, int dim_z)
	{
		vol_mat = new3D(dim_z, dim_y, dim_x);
		this->dim_x = dim_x;
		this->dim_y = dim_y;
		this->dim_z = dim_z;
		size = dim_z * dim_y * dim_x;
	}

	Image3D::Image3D(std::string file_path)
	{
		load(file_path);
	}

	void Image3D::loadBin(std::string file_path)
	{
		if (vol_mat != nullptr)
		{
			delete3D(vol_mat);
		}

		std::ifstream file_in;
		file_in.open(file_path, std::ios::in | std::ios::binary);

		if (!file_in.is_open())
		{
			std::cerr << "Failed to open bin file: " << file_path << std::endl;
		}

		//get the length of data
		file_in.seekg(0, file_in.end);
		int file_length = file_in.tellg();
		file_in.seekg(0, file_in.beg);
		int data_length = file_length - sizeof(int) * 3;

		//head information is an array of int[3]: dimension of x, y, and z
		int img_dimension[3];
		file_in.read((char*)img_dimension, sizeof(int) * 3);
		dim_x = img_dimension[0];
		dim_y = img_dimension[1];
		dim_z = img_dimension[2];
		size = dim_z * dim_y * dim_x;

		//create a 3D matrix and fill it with the data (float) in binary file
		vol_mat = new3D(dim_z, dim_y, dim_x);
		file_in.read((char*)**vol_mat, sizeof(float) * size);

		file_in.close();
	}

	void Image3D::loadTiff(std::string file_path)
	{
		if (vol_mat != nullptr)
		{
			delete3D(vol_mat);
		}

		//read a tiff image consisting of multiple pages and store it in a vector of cv::Mat
		std::vector<cv::Mat> tiff_mat;
		if (!cv::imreadmulti(file_path, tiff_mat, cv::IMREAD_GRAYSCALE))
		{
			std::cerr << "Fail to load multi-page tiff: " + file_path << std::endl;
		}

		//get the three dimensions of 3D image
		dim_x = tiff_mat[0].cols;
		dim_y = tiff_mat[0].rows;
		dim_z = (int)tiff_mat.size();
		size = dim_z * dim_y * dim_x;

		//create a 3D matrix and fill it with the data ifnTIFF
		vol_mat = new3D(dim_z, dim_y, dim_x);

#pragma omp parallel for
		for (int i = 0; i < dim_z; i++)
		{
			for (int j = 0; j < dim_y; j++)
			{
				for (int k = 0; k < dim_x; k++) {
					vol_mat[i][j][k] = (float)tiff_mat[i].at<uchar>(j, k);
				}
			}
		}
	}

	void Image3D::load(std::string file_path)
	{
		this->file_path = file_path;

		//check if the file is a bin or tiff
		size_t dot_pos = file_path.find_last_of(".");
		std::string file_ext = file_path.substr(dot_pos + 1);
		if (file_ext == "bin" || file_ext == "BIN")
		{
			loadBin(file_path);
		}
		else if (file_ext == "tif" || file_ext == "TIF" || file_ext == "tiff" || file_ext == "TIFF")
		{
			loadTiff(file_path);
		}
		else
		{
			std::cerr << "Not binary file or multi-page tiff" << std::endl;
		}
	}

	void Image3D::release()
	{
		if (vol_mat != nullptr)
		{
			delete3D(vol_mat);
		}
	}


	//colorful image 2D
	ColorfulImage2D::ColorfulImage2D(int width, int height)
	{
		cv_mat = cv::Mat::zeros(height, width, CV_8UC3);
		b_eg_mat = Eigen::MatrixXf::Zero(height, width);
		g_eg_mat = Eigen::MatrixXf::Zero(height, width);
		r_eg_mat = Eigen::MatrixXf::Zero(height, width);
		this->width = width;
		this->height = height;
		size = height * width;
	}

	ColorfulImage2D::ColorfulImage2D(std::string file_path)
	{
		load(file_path);
	}

	void ColorfulImage2D::load(std::string file_path)
	{
		cv_mat = cv::imread(file_path, cv::IMREAD_COLOR);

		if (!cv_mat.data)
		{
			throw std::string("Fail to load file: " + file_path);
		}

		this->file_path = file_path;

		//clear channels
		std::vector<cv::Mat>().swap(cv_channels);
		//split the three color channels, following the order of B, G, and R
		split(cv_mat, cv_channels);

		if (width != cv_mat.cols || height != cv_mat.rows)
		{
			width = cv_mat.cols;
			height = cv_mat.rows;
			size = height * width;
			b_eg_mat.resize(height, width);
			g_eg_mat.resize(height, width);
			r_eg_mat.resize(height, width);
		}

		cv::cv2eigen(cv_channels[0], b_eg_mat);
		cv::cv2eigen(cv_channels[1], g_eg_mat);
		cv::cv2eigen(cv_channels[2], r_eg_mat);
	}

}//namespace opencorr

