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

#ifndef _FFTCC_H_
#define _FFTCC_H_

#include <array>
#include <vector>
#include "fftw3.h"

#include "oc_dic.h"

namespace opencorr
{
	class FFTW
	{
	public:
		float* ref_subset;
		float* tar_subset;
		float* zncc;
		fftwf_complex* ref_freq;
		fftwf_complex* tar_freq;
		fftwf_complex* zncc_freq;
		fftwf_plan ref_plan;
		fftwf_plan tar_plan;
		fftwf_plan zncc_plan;

		static std::unique_ptr<FFTW> allocate(int subset_radius_x, int subset_radius_y);
		static std::unique_ptr<FFTW> allocate(int subset_radius_x, int subset_radius_y, int subset_radius_z);

		static void release(std::unique_ptr<FFTW>& instance);

		static void update(std::unique_ptr<FFTW>& instance, int subset_radius_x, int subset_radius_y);
		static void update(std::unique_ptr<FFTW>&, int subset_radius_x, int subset_radius_y, int subset_radius_z);
	};


	//the 2D part of module is the implementation of
	//Z. Jiang et al, Optics and Lasers in Engineering (2015) 65: 93-102.
	//https://doi.org/10.1016/j.optlaseng.2014.06.011

	class FFTCC2D : public DIC
	{
	private:
		struct PreparedRef2D
		{
			int x = 0;
			int y = 0;
			bool valid = false;
			float norm = 0.f;
			std::vector<std::array<float, 2>> ref_freq;
		};

		std::vector<std::unique_ptr<FFTW>> instance_pool; //pool of FFTW instances for multi-thread processing
		std::vector<PreparedRef2D> prepared_ref_;
		bool prepared_ref_ready_ = false;
		int allocated_radius_ = 0;
		void validateGeometry() const;
		std::unique_ptr<FFTW>& getInstance(int tid); //get an instance according to the number of current thread id
		bool preparedLayoutMatches(const std::vector<POI2D>& poi_queue) const;
		void computePreparedRef(POI2D* poi, int cache_index);
		void ensureReferenceCache();

	public:
		FFTCC2D(int subset_radius_x, int subset_radius_y, int thread_number);
		~FFTCC2D();

		// Content-change notifications, including same-address image reloads.
		void setImages(Image2D& ref, Image2D& tar) override;
		void setRefImage(Image2D& ref);
		void setTarImage(Image2D& tar);
		bool hasPreparedReference(const Image2D& ref) const noexcept
		{ return ref_img == &ref && prepared_ref_ready_
		         && subset_radius_x == allocated_radius_ && subset_radius_y == allocated_radius_; }

		void prepare();
		void prepareRef();
		void prepareRef(const std::vector<POI2D>& poi_queue);
		void prepareTar();

		void compute(POI2D* poi);
		void compute(std::vector<POI2D>& poi_queue);
		// Indexed calls require a previously established layout; reference changes
		// lazily rebuild it. Invalid indices or coordinates throw, never fall back.
		void compute(std::vector<POI2D>& poi_queue, const std::vector<size_t>& cache_indices);
	};


	//the 3D part of module is the implementation of
	//T. Wang et al, Experimental Mechanics (2016) 56(2): 297-309.
	//https://doi.org/10.1007/s11340-015-0091-4

	class FFTCC3D : public DVC
	{
	private:
		std::vector<std::unique_ptr<FFTW>> instance_pool; //pool of FFTW instances for multi-thread processing
		std::unique_ptr<FFTW>& getInstance(int tid); //get an instance according to the number of current thread id

	public:
		FFTCC3D(int subset_radius_x, int subset_radius_y, int subset_radius_z, int thread_number);
		~FFTCC3D();

		void prepare();

		void compute(POI3D* poi);
		void compute(std::vector<POI3D>& poi_queue);
	};

}//namespace opencorr

#endif //_FFTCC_H_
