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

#pragma once

#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>

#include "oc_array.h"

namespace opencorr
{
	enum class ImageInputDepthMode
	{
		Compat8,
		Depth12,
		Depth16,
		Auto
	};

	enum class ImageInputAlignment
	{
		Right,
		Left,
		Auto
	};

	enum class ImageInputNormalize
	{
		None,
		To8,
		To16,
		Unit
	};

	struct ImageLoadOptions
	{
		ImageInputDepthMode depth = ImageInputDepthMode::Compat8;
		ImageInputAlignment alignment = ImageInputAlignment::Auto;
		ImageInputNormalize normalize = ImageInputNormalize::None;
	};

	struct ImageLoadInfo
	{
		ImageInputDepthMode requested_depth = ImageInputDepthMode::Compat8;
		ImageInputDepthMode effective_depth = ImageInputDepthMode::Compat8;
		ImageInputAlignment requested_alignment = ImageInputAlignment::Auto;
		ImageInputAlignment effective_alignment = ImageInputAlignment::Auto;
		ImageInputNormalize normalize = ImageInputNormalize::None;
		int imread_flags = cv::IMREAD_GRAYSCALE;
		bool used_anydepth = false;
		int source_depth = -1;
		int source_channels = 0;
		double source_min = 0.0;
		double source_max = 0.0;
		int output_depth = -1;
		double output_min = 0.0;
		double output_max = 0.0;
		double output_scale = 1.0;
	};

	const char* imageInputDepthModeName(ImageInputDepthMode depth);
	const char* imageInputAlignmentName(ImageInputAlignment alignment);
	const char* imageInputNormalizeName(ImageInputNormalize normalize);
	std::string imageCvDepthName(int depth);
	cv::Mat loadImage2DMat(std::string file_path, const ImageLoadOptions& options,
		ImageLoadInfo* info = nullptr);

	class Image2D
	{
	public:
		int height, width;
		unsigned int size;

		std::string file_path;

		cv::Mat cv_mat;
		Eigen::MatrixXf eg_mat;
		ImageLoadInfo load_info;

		Image2D(int width, int height);
		Image2D(std::string file_path);
		// user-declared dtor suppresses implicit move; restore it explicitly so
		// `tar_img = std::move(*preloaded.image)` is a pointer swap, not a deep copy
		Image2D(Image2D&&) = default;
		Image2D& operator=(Image2D&&) = default;
		~Image2D() = default;

		void load(std::string file_path);
		void loadWithOptions(std::string file_path, const ImageLoadOptions& options,
			ImageLoadInfo* info = nullptr);
	};

	class Image3D
	{
	public:
		int dim_x, dim_y, dim_z;
		unsigned long size;

		std::string file_path;

		float*** vol_mat = nullptr;

		Image3D(int dim_x, int dim_y, int dim_z);
		Image3D(std::string file_path);
		~Image3D() = default;

		void loadBin(std::string file_path);
		void loadTiff(std::string file_path);
		void load(std::string file_path);

		void release();
	};

	//class made to process colorful image
	class ColorfulImage2D
	{
	public:
		int height, width;
		unsigned int size;

		std::string file_path;

		cv::Mat cv_mat;
		std::vector<cv::Mat> cv_channels;
		Eigen::MatrixXf b_eg_mat, g_eg_mat, r_eg_mat;

		ColorfulImage2D(int width, int height);
		ColorfulImage2D(std::string file_path);
		~ColorfulImage2D() = default;

		void load(std::string file_path);
	};

}//namespace opencorr

#endif //_IMAGE_H_
