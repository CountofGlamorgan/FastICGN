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

#include <omp.h>

#include "oc_fftcc.h"
#include <stdexcept>
#include <cmath>

namespace opencorr
{
	std::unique_ptr<FFTW> FFTW::allocate(int subset_radius_x, int subset_radius_y)
	{
		int width = 2 * subset_radius_x;
		int height = 2 * subset_radius_y;
		int buffer_length = width * (subset_radius_y + 1);
		unsigned int subset_size = width * height;

		std::unique_ptr<FFTW> FFTW_instance = std::make_unique<FFTW>();

		FFTW_instance->ref_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		FFTW_instance->tar_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		FFTW_instance->zncc_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);

		FFTW_instance->ref_subset = new float[subset_size];
		FFTW_instance->tar_subset = new float[subset_size];
		FFTW_instance->zncc = new float[subset_size];

#pragma omp critical 
		{
			FFTW_instance->ref_plan = fftwf_plan_dft_r2c_2d(width, height, FFTW_instance->ref_subset, FFTW_instance->ref_freq, FFTW_ESTIMATE);
			FFTW_instance->tar_plan = fftwf_plan_dft_r2c_2d(width, height, FFTW_instance->tar_subset, FFTW_instance->tar_freq, FFTW_ESTIMATE);
			FFTW_instance->zncc_plan = fftwf_plan_dft_c2r_2d(width, height, FFTW_instance->zncc_freq, FFTW_instance->zncc, FFTW_ESTIMATE);
		}

		return FFTW_instance;
	}

	std::unique_ptr<FFTW> FFTW::allocate(int subset_radius_x, int subset_radius_y, int subset_radius_z)
	{
		int dim_x = 2 * subset_radius_x;
		int dim_y = 2 * subset_radius_y;
		int dim_z = 2 * subset_radius_z;
		int buffer_length = dim_x * dim_y * (subset_radius_z + 1);
		unsigned int subset_size = dim_x * dim_y * dim_z;

		std::unique_ptr<FFTW> FFTW_instance = std::make_unique<FFTW>();

		FFTW_instance->ref_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		FFTW_instance->tar_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		FFTW_instance->zncc_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);

		FFTW_instance->ref_subset = new float[subset_size];
		FFTW_instance->tar_subset = new float[subset_size];
		FFTW_instance->zncc = new float[subset_size];

#pragma omp critical 
		{
			FFTW_instance->ref_plan = fftwf_plan_dft_r2c_3d(dim_x, dim_y, dim_z, FFTW_instance->ref_subset, FFTW_instance->ref_freq, FFTW_ESTIMATE);
			FFTW_instance->tar_plan = fftwf_plan_dft_r2c_3d(dim_x, dim_y, dim_z, FFTW_instance->tar_subset, FFTW_instance->tar_freq, FFTW_ESTIMATE);
			FFTW_instance->zncc_plan = fftwf_plan_dft_c2r_3d(dim_x, dim_y, dim_z, FFTW_instance->zncc_freq, FFTW_instance->zncc, FFTW_ESTIMATE);
		}

		return FFTW_instance;
	}

	void FFTW::release(std::unique_ptr<FFTW>& instance)
	{
		delete[] instance->ref_subset;
		delete[] instance->tar_subset;
		delete[] instance->zncc;
		fftw_free(instance->ref_freq);
		fftw_free(instance->tar_freq);
		fftw_free(instance->zncc_freq);
		fftwf_destroy_plan(instance->ref_plan);
		fftwf_destroy_plan(instance->tar_plan);
		fftwf_destroy_plan(instance->zncc_plan);
	}

	void FFTW::update(std::unique_ptr<FFTW>& instance, int subset_radius_x, int subset_radius_y)
	{
		release(instance);

		int width = 2 * subset_radius_x;
		int height = 2 * subset_radius_y;
		int buffer_length = width * (subset_radius_y + 1);
		unsigned int subset_size = width * height;

		instance->ref_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		instance->tar_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		instance->zncc_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);

		instance->ref_subset = new float[subset_size];
		instance->tar_subset = new float[subset_size];
		instance->zncc = new float[subset_size];

#pragma omp critical 
		{
			instance->ref_plan = fftwf_plan_dft_r2c_2d(width, height, instance->ref_subset, instance->ref_freq, FFTW_ESTIMATE);
			instance->tar_plan = fftwf_plan_dft_r2c_2d(width, height, instance->tar_subset, instance->tar_freq, FFTW_ESTIMATE);
			instance->zncc_plan = fftwf_plan_dft_c2r_2d(width, height, instance->zncc_freq, instance->zncc, FFTW_ESTIMATE);
		}
	}

	void FFTW::update(std::unique_ptr<FFTW>& instance, int subset_radius_x, int subset_radius_y, int subset_radius_z)
	{
		release(instance);

		int dim_x = 2 * subset_radius_x;
		int dim_y = 2 * subset_radius_y;
		int dim_z = 2 * subset_radius_z;
		int buffer_length = dim_x * dim_y * (subset_radius_z + 1);
		unsigned int subset_size = dim_x * dim_y * dim_z;

		instance->ref_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		instance->tar_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);
		instance->zncc_freq = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * buffer_length);

		instance->ref_subset = new float[subset_size];
		instance->tar_subset = new float[subset_size];
		instance->zncc = new float[subset_size];

#pragma omp critical 
		{
			instance->ref_plan = fftwf_plan_dft_r2c_3d(dim_x, dim_y, dim_z, instance->ref_subset, instance->ref_freq, FFTW_ESTIMATE);
			instance->tar_plan = fftwf_plan_dft_r2c_3d(dim_x, dim_y, dim_z, instance->tar_subset, instance->tar_freq, FFTW_ESTIMATE);
			instance->zncc_plan = fftwf_plan_dft_c2r_3d(dim_x, dim_y, dim_z, instance->zncc_freq, instance->zncc, FFTW_ESTIMATE);
		}
	}

	//FFT accelerated cross correlation 2D
	std::unique_ptr<FFTW>& FFTCC2D::getInstance(int tid)
	{
		if (tid >= (int)instance_pool.size())
		{
			throw std::string("CPU thread ID over limit");
		}

		return instance_pool[tid];
	}

	FFTCC2D::FFTCC2D(int subset_radius_x, int subset_radius_y, int thread_number)
	{
		if (subset_radius_x <= 0 || subset_radius_x != subset_radius_y || subset_radius_x > 23170)
			throw std::invalid_argument("FFTCC2D supports positive square radii only (1..23170); rectangular FFTs are unsupported");
		if (thread_number < 1) throw std::invalid_argument("FFTCC2D thread count must be positive");
		allocated_radius_ = subset_radius_x;
		this->subset_radius_x = subset_radius_x;
		this->subset_radius_y = subset_radius_y;
		this->thread_number = thread_number;

		instance_pool.resize(thread_number);
#pragma omp parallel for
		for (int i = 0; i < thread_number; i++)
		{
			instance_pool[i] = FFTW::allocate(subset_radius_x, subset_radius_y);
		}
	}

	FFTCC2D::~FFTCC2D()
	{
		for (auto& instance : instance_pool)
		{
			FFTW::release(instance);
			instance.reset();
		}
		std::vector<std::unique_ptr<FFTW >>().swap(instance_pool);
	}

	void FFTCC2D::setImages(Image2D& ref, Image2D& tar)
	{
		setRefImage(ref);
		setTarImage(tar);
	}

	void FFTCC2D::setRefImage(Image2D& ref)
	{
		ref_img = &ref;
		prepared_ref_ready_ = false; // retain layout for lazy spectral rebuild
	}

	void FFTCC2D::setTarImage(Image2D& tar)
	{
		tar_img = &tar; // FFTW reads the current target directly; no target cache
	}

	void FFTCC2D::ensureReferenceCache()
	{
		if (prepared_ref_ready_) return;
		std::vector<POI2D> layout;
		layout.reserve(prepared_ref_.size());
		for (const auto& poi : prepared_ref_)
			layout.emplace_back(static_cast<float>(poi.x), static_cast<float>(poi.y));
		prepareRef(layout);
	}

	void FFTCC2D::validateGeometry() const
	{
		if (subset_radius_x != allocated_radius_ || subset_radius_y != allocated_radius_)
			throw std::invalid_argument("FFTCC2D subset geometry is fixed at construction; create a new solver");
	}

	void FFTCC2D::prepare()
	{
		prepareRef();
		prepareTar();
	}

	void FFTCC2D::prepareRef()
	{
		validateGeometry();
		prepared_ref_.clear();
		prepared_ref_ready_ = false;
	}

	bool FFTCC2D::preparedLayoutMatches(const std::vector<POI2D>& poi_queue) const
	{
		if (prepared_ref_.size() != poi_queue.size())
		{
			return false;
		}

		for (size_t i = 0; i < poi_queue.size(); ++i)
		{
			if (prepared_ref_[i].x != static_cast<int>(poi_queue[i].x)
				|| prepared_ref_[i].y != static_cast<int>(poi_queue[i].y))
			{
				return false;
			}
		}
		return true;
	}

	void FFTCC2D::prepareRef(const std::vector<POI2D>& poi_queue)
	{
		validateGeometry();
		if (ref_img == nullptr)
		{
			throw std::string("Reference image must be set before FFTCC2D::prepareRef");
		}

		int subset_width = subset_radius_x * 2;
		int subset_height = subset_radius_y * 2;
		int subset_size = subset_width * subset_height;
		int buffer_length = subset_width * (subset_radius_y + 1);

		prepared_ref_ready_ = false;
		prepared_ref_.assign(poi_queue.size(), PreparedRef2D());
		//num_threads pinned as in upstream 0a795a0 ("Pin num_threads on parallel-for loops
		//that index a fixed-size instance pool").  The body picks its FFTW instance with
		//getInstance(omp_get_thread_num()), which throws once the thread id leaves the
		//thread_number-sized instance_pool.  This loop is one of the three parallel regions
		//added by this project, so upstream's patch does not reach it.
#pragma omp parallel for num_threads(thread_number)
		for (int i = 0; i < static_cast<int>(poi_queue.size()); ++i)
		{
			std::unique_ptr<FFTW>& fftcc = getInstance(omp_get_thread_num());
			PreparedRef2D& prepared = prepared_ref_[i];
			prepared.x = static_cast<int>(poi_queue[i].x);
			prepared.y = static_cast<int>(poi_queue[i].y);
			prepared.ref_freq.assign(buffer_length, { 0.f, 0.f });

			if (prepared.x < subset_radius_x || prepared.x >= ref_img->width - subset_radius_x
				|| prepared.y < subset_radius_y || prepared.y >= ref_img->height - subset_radius_y)
			{
				continue;
			}

			float ref_mean = 0.f;
			for (int r = 0; r < subset_height; r++)
			{
				for (int c = 0; c < subset_width; c++)
				{
					int x = prepared.x + c - subset_radius_x;
					int y = prepared.y + r - subset_radius_y;
					float value = ref_img->eg_mat(y, x);
					fftcc->ref_subset[r * subset_width + c] = value;
					ref_mean += value;
				}
			}
			ref_mean /= subset_size;

			float ref_norm = 0.f;
			for (int idx = 0; idx < subset_size; ++idx)
			{
				fftcc->ref_subset[idx] -= ref_mean;
				ref_norm += fftcc->ref_subset[idx] * fftcc->ref_subset[idx];
			}

			fftwf_execute(fftcc->ref_plan);
			for (int n = 0; n < buffer_length; ++n)
			{
				prepared.ref_freq[n][0] = fftcc->ref_freq[n][0];
				prepared.ref_freq[n][1] = fftcc->ref_freq[n][1];
			}
			prepared.norm = ref_norm;
			prepared.valid = true;
		}

		prepared_ref_ready_ = true;
	}

	void FFTCC2D::prepareTar() { validateGeometry(); }

	void FFTCC2D::computePreparedRef(POI2D* poi, int cache_index)
	{
		std::unique_ptr<FFTW>& fftcc = getInstance(omp_get_thread_num());

		int subset_width = subset_radius_x * 2;
		int subset_height = subset_radius_y * 2;
		int subset_size = subset_width * subset_height;
		int buffer_length = subset_width * (subset_radius_y + 1);

		Point2D initial_displacement(poi->deformation.u, poi->deformation.v);
		const PreparedRef2D& prepared = prepared_ref_[cache_index];
		if (!prepared.valid
			|| int(poi->x + initial_displacement.x) < subset_radius_x || int(poi->x + initial_displacement.x) >= ref_img->width - subset_radius_x
			|| int(poi->y + initial_displacement.y) < subset_radius_y || int(poi->y + initial_displacement.y) >= ref_img->height - subset_radius_y)
		{
			return;
		}

		float tar_mean = 0.f;
		float tar_norm = 0.f;
		for (int r = 0; r < subset_height; r++)
		{
			for (int c = 0; c < subset_width; c++)
			{
				Point2D ref_point(poi->x + c - subset_radius_x, poi->y + r - subset_radius_y);
				Point2D tar_point = ref_point + initial_displacement;
				float value = tar_img->eg_mat((int)tar_point.y, (int)tar_point.x);
				fftcc->tar_subset[r * subset_width + c] = value;
				tar_mean += value;
			}
		}
		tar_mean /= subset_size;

		for (int i = 0; i < subset_size; i++)
		{
			fftcc->tar_subset[i] -= tar_mean;
			tar_norm += fftcc->tar_subset[i] * fftcc->tar_subset[i];
		}

		fftwf_execute(fftcc->tar_plan);

		for (int n = 0; n < buffer_length; n++)
		{
			fftcc->zncc_freq[n][0] = (prepared.ref_freq[n][0] * fftcc->tar_freq[n][0]) + (prepared.ref_freq[n][1] * fftcc->tar_freq[n][1]);
			fftcc->zncc_freq[n][1] = (prepared.ref_freq[n][0] * fftcc->tar_freq[n][1]) - (prepared.ref_freq[n][1] * fftcc->tar_freq[n][0]);
		}

		fftwf_execute(fftcc->zncc_plan);

		float max_zncc = -2.f;
		int max_zncc_index = 0;
		for (int i = 0; i < subset_size; i++)
		{
			if (fftcc->zncc[i] > max_zncc)
			{
				max_zncc = fftcc->zncc[i];
				max_zncc_index = i;
			}
		}
		int local_displacement_u = max_zncc_index % subset_width;
		int local_displacement_v = max_zncc_index / subset_width;

		if (local_displacement_u > subset_radius_x)
		{
			local_displacement_u -= subset_width;
		}
		if (local_displacement_v > subset_radius_y)
		{
			local_displacement_v -= subset_height;
		}

		poi->deformation.u = (float)local_displacement_u + initial_displacement.x;
		poi->deformation.v = (float)local_displacement_v + initial_displacement.y;

		poi->result.u0 = initial_displacement.x;
		poi->result.v0 = initial_displacement.y;
		poi->result.zncc = max_zncc / (sqrt(prepared.norm * tar_norm) * subset_size);
	}

	void FFTCC2D::compute(POI2D* poi)
	{
		validateGeometry();
		if (!poi || !ref_img || !tar_img) throw std::logic_error("FFTCC2D compute requires images and a POI");
		//set instance w.r.t. thread id 
		std::unique_ptr<FFTW>& fftcc = getInstance(omp_get_thread_num());

		int subset_width = subset_radius_x * 2;
		int subset_height = subset_radius_y * 2;
		int subset_size = subset_width * subset_height;

		//set initial guess of displacement
		Point2D initial_displacement(poi->deformation.u, poi->deformation.v);

		//check if the position of POI is too close to the boundary
		if ((int)poi->x < subset_radius_x || (int)poi->x >= ref_img->width - subset_radius_x
			|| (int)poi->y < subset_radius_y || (int)poi->y >= ref_img->height - subset_radius_y
			|| int(poi->x + initial_displacement.x) < subset_radius_x || int(poi->x + initial_displacement.x) >= ref_img->width - subset_radius_x
			|| int(poi->y + initial_displacement.y) < subset_radius_y || int(poi->y + initial_displacement.y) >= ref_img->height - subset_radius_y)
		{
			return;
		}

		//initialize mean and norm in subsets
		float ref_mean = 0.f;
		float tar_mean = 0.f;
		float ref_norm = 0.f;
		float tar_norm = 0.f;

		for (int r = 0; r < subset_height; r++)
		{
			for (int c = 0; c < subset_width; c++)
			{
				//fill reference subset
				Point2D ref_point(poi->x + c - subset_radius_x, poi->y + r - subset_radius_y);
				float value = ref_img->eg_mat((int)ref_point.y, (int)ref_point.x);
				fftcc->ref_subset[r * subset_width + c] = value;
				ref_mean += value;

				//fill the target subset with initial guess of displacement
				Point2D tar_point = ref_point + initial_displacement;
				value = tar_img->eg_mat((int)tar_point.y, (int)tar_point.x);
				fftcc->tar_subset[r * subset_width + c] = value;
				tar_mean += value;
			}
		}
		ref_mean /= subset_size;
		tar_mean /= subset_size;

		//zero-mean operation of gray-scale values in the two subsets
		for (int i = 0; i < subset_size; i++)
		{
			fftcc->ref_subset[i] -= ref_mean;
			fftcc->tar_subset[i] -= tar_mean;
			ref_norm += fftcc->ref_subset[i] * fftcc->ref_subset[i];
			tar_norm += fftcc->tar_subset[i] * fftcc->tar_subset[i];
		}

		fftwf_execute(fftcc->ref_plan);
		fftwf_execute(fftcc->tar_plan);

		int buffer_length = subset_width * (subset_radius_y + 1);
		for (int n = 0; n < buffer_length; n++)
		{
			fftcc->zncc_freq[n][0] = (fftcc->ref_freq[n][0] * fftcc->tar_freq[n][0]) + (fftcc->ref_freq[n][1] * fftcc->tar_freq[n][1]);
			fftcc->zncc_freq[n][1] = (fftcc->ref_freq[n][0] * fftcc->tar_freq[n][1]) - (fftcc->ref_freq[n][1] * fftcc->tar_freq[n][0]);
		}

		fftwf_execute(fftcc->zncc_plan);

		//search for max ZCC
		float max_zncc = -2.f;
		int max_zncc_index = 0;
		for (int i = 0; i < subset_size; i++)
		{
			if (fftcc->zncc[i] > max_zncc)
			{
				max_zncc = fftcc->zncc[i];
				max_zncc_index = i;
			}
		}
		int local_displacement_u = max_zncc_index % subset_width;
		int local_displacement_v = max_zncc_index / subset_width;

		if (local_displacement_u > subset_radius_x)
		{
			local_displacement_u -= subset_width;
		}
		if (local_displacement_v > subset_radius_y)
		{
			local_displacement_v -= subset_height;
		}

		//store the final results
		poi->deformation.u = (float)local_displacement_u + initial_displacement.x;
		poi->deformation.v = (float)local_displacement_v + initial_displacement.y;

		poi->result.u0 = initial_displacement.x;
		poi->result.v0 = initial_displacement.y;
		poi->result.zncc = max_zncc / (sqrt(ref_norm * tar_norm) * subset_size); //convert ZCC to ZNCC
	}

	void FFTCC2D::compute(std::vector<POI2D>& poi_queue, const std::vector<size_t>& cache_indices)
	{
		validateGeometry();
		if (ref_img == nullptr || tar_img == nullptr)
			throw std::logic_error("FFTCC2D images must be set before compute");
		if (cache_indices.size() != poi_queue.size())
			throw std::invalid_argument("FFTCC2D::compute requires one cache index per POI");
		if (poi_queue.empty()) return;
		if (prepared_ref_.empty())
			throw std::logic_error("FFTCC2D indexed compute requires a prepared reference layout");
		for (size_t i = 0; i < poi_queue.size(); ++i)
		{
			const size_t index = cache_indices[i];
			if (index >= prepared_ref_.size())
				throw std::out_of_range("FFTCC2D reference cache index out of range");
			if (!std::isfinite(poi_queue[i].x) || !std::isfinite(poi_queue[i].y)
				|| prepared_ref_[index].x != static_cast<int>(poi_queue[i].x)
				|| prepared_ref_[index].y != static_cast<int>(poi_queue[i].y))
				throw std::invalid_argument("FFTCC2D reference cache index does not match POI coordinates");
		}
		ensureReferenceCache();
		//num_threads pinned as in upstream 0a795a0 ("Pin num_threads on parallel-for loops
		//that index a fixed-size instance pool").  The body picks its FFTW instance with
		//getInstance(omp_get_thread_num()), which throws once the thread id leaves the
		//thread_number-sized instance_pool.  This loop is one of the three parallel regions
		//added by this project, so upstream's patch does not reach it.
#pragma omp parallel for num_threads(thread_number)
		for (int i = 0; i < static_cast<int>(poi_queue.size()); ++i)
			computePreparedRef(&poi_queue[i], static_cast<int>(cache_indices[i]));
	}

	void FFTCC2D::compute(std::vector<POI2D>& poi_queue)
	{
		validateGeometry();
		if (ref_img == nullptr || tar_img == nullptr)
			throw std::logic_error("FFTCC2D images must be set before compute");
		if (poi_queue.empty()) return;
		auto queue_length = poi_queue.size();
		if (preparedLayoutMatches(poi_queue))
		{
			ensureReferenceCache();
			//num_threads pinned as in upstream 0a795a0 ("Pin num_threads on parallel-for loops
			//that index a fixed-size instance pool").  The body picks its FFTW instance with
			//getInstance(omp_get_thread_num()), which throws once the thread id leaves the
			//thread_number-sized instance_pool.  This loop is one of the three parallel regions
			//added by this project, so upstream's patch does not reach it.
#pragma omp parallel for num_threads(thread_number)
			for (int i = 0; i < queue_length; i++)
			{
				computePreparedRef(&poi_queue[i], i);
			}
			return;
		}

#pragma omp parallel for num_threads(thread_number)
		for (int i = 0; i < queue_length; i++)
		{
			compute(&poi_queue[i]);
		}
	}



	//FFT accelerated cross correlation 3D
	std::unique_ptr<FFTW>& FFTCC3D::getInstance(int tid)
	{
		if (tid >= (int)instance_pool.size())
		{
			throw std::string("CPU thread ID over limit");
		}

		return instance_pool[tid];
	}

	FFTCC3D::FFTCC3D(int subset_radius_x, int subset_radius_y, int subset_radius_z, int thread_number)
	{
		this->subset_radius_x = subset_radius_x;
		this->subset_radius_y = subset_radius_y;
		this->subset_radius_z = subset_radius_z;
		this->thread_number = thread_number;

		instance_pool.resize(thread_number);
#pragma omp parallel for
		for (int i = 0; i < thread_number; i++)
		{
			instance_pool[i] = FFTW::allocate(subset_radius_x, subset_radius_y, subset_radius_z);
		}
	}

	FFTCC3D::~FFTCC3D()
	{
		for (auto& instance : instance_pool)
		{
			FFTW::release(instance);
			instance.reset();
		}
		std::vector<std::unique_ptr<FFTW>>().swap(instance_pool);
	}

	void FFTCC3D::prepare() {}

	void FFTCC3D::compute(POI3D* poi)
	{
		//set instance w.r.t. thread id 
		std::unique_ptr<FFTW>& fftcc = getInstance(omp_get_thread_num());

		int subset_dim_x = subset_radius_x * 2;
		int subset_dim_y = subset_radius_y * 2;
		int subset_dim_z = subset_radius_z * 2;
		int subset_size = subset_dim_x * subset_dim_y * subset_dim_z;

		//set initial guess of displacement
		Point3D initial_displacement(poi->deformation.u, poi->deformation.v, poi->deformation.w);

		//initialize mean and norm in subsets
		float ref_mean = 0.f;
		float tar_mean = 0.f;
		float ref_norm = 0.f;
		float tar_norm = 0.f;

		for (int i = 0; i < subset_dim_z; i++)
		{
			for (int j = 0; j < subset_dim_y; j++)
			{
				for (int k = 0; k < subset_dim_x; k++)
				{
					//fill the reference subset
					Point3D ref_point(poi->x + k - subset_radius_x, poi->y + j - subset_radius_y, poi->z + i - subset_radius_z);
					float value = ref_img->vol_mat[(int)ref_point.z][(int)ref_point.y][(int)ref_point.x];
					fftcc->ref_subset[(i * subset_dim_y + j) * subset_dim_x + k] = value;
					ref_mean += value;

					//fill the target subset with initial guess of displacement
					Point3D tar_point = ref_point + initial_displacement;
					value = tar_img->vol_mat[(int)tar_point.z][(int)tar_point.y][(int)tar_point.x];
					fftcc->tar_subset[(i * subset_dim_y + j) * subset_dim_x + k] = value;
					tar_mean += value;
				}
			}
		}
		ref_mean /= subset_size;
		tar_mean /= subset_size;

		//zero-mean operation of gray-scale values in the two subsets
		for (int i = 0; i < subset_size; i++)
		{
			fftcc->ref_subset[i] -= ref_mean;
			fftcc->tar_subset[i] -= tar_mean;
			ref_norm += fftcc->ref_subset[i] * fftcc->ref_subset[i];
			tar_norm += fftcc->tar_subset[i] * fftcc->tar_subset[i];
		}

		fftwf_execute(fftcc->ref_plan);
		fftwf_execute(fftcc->tar_plan);

		unsigned int buffer_length = subset_dim_x * subset_dim_y * (subset_radius_z + 1);
		for (unsigned int n = 0; n < buffer_length; n++)
		{
			fftcc->zncc_freq[n][0] = (fftcc->ref_freq[n][0] * fftcc->tar_freq[n][0]) + (fftcc->ref_freq[n][1] * fftcc->tar_freq[n][1]);
			fftcc->zncc_freq[n][1] = (fftcc->ref_freq[n][0] * fftcc->tar_freq[n][1]) - (fftcc->ref_freq[n][1] * fftcc->tar_freq[n][0]);
		}

		fftwf_execute(fftcc->zncc_plan);

		//search for max ZCC
		float max_zncc = -2.f;
		int max_zncc_index = 0;
		for (int i = 0; i < subset_size; i++)
		{
			if (fftcc->zncc[i] > max_zncc)
			{
				max_zncc = fftcc->zncc[i];
				max_zncc_index = i;
			}
		}
		int local_displacement_u = max_zncc_index % subset_dim_x;
		int local_displacement_v = (max_zncc_index / subset_dim_x) % subset_dim_y;
		int local_displacement_w = max_zncc_index / (subset_dim_x * subset_dim_y);

		if (local_displacement_u > subset_radius_x)
		{
			local_displacement_u -= subset_dim_x;
		}
		if (local_displacement_v > subset_radius_y)
		{
			local_displacement_v -= subset_dim_y;
		}
		if (local_displacement_w > subset_radius_z)
		{
			local_displacement_w -= subset_dim_z;
		}

		//store the final results
		poi->deformation.u = (float)local_displacement_u + initial_displacement.x;
		poi->deformation.v = (float)local_displacement_v + initial_displacement.y;
		poi->deformation.w = (float)local_displacement_w + initial_displacement.z;

		poi->result.u0 = initial_displacement.x;
		poi->result.v0 = initial_displacement.y;
		poi->result.w0 = initial_displacement.z;
		poi->result.zncc = max_zncc / (sqrt(ref_norm * tar_norm) * subset_size); //convert ZCC to ZNCC
	}

	void FFTCC3D::compute(std::vector<POI3D>& poi_queue)
	{
		auto queue_length = poi_queue.size();
#pragma omp parallel for num_threads(thread_number)
		for (int i = 0; i < queue_length; i++)
		{
			compute(&poi_queue[i]);
		}
	}

}//namespace opencorr
