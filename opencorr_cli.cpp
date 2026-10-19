/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/*
 * OpenCorr CLI wrapper — general-purpose command-line DIC tool
 * Usage: opencorr_cli <ref_image> <tar_image> <subset_radius> <grid_space> <output_csv>
 *        [--roi x,y,w,h] [--line y1,y2] [--threads N]
 *
 * Performs FFTCC + ICGN1 (default) on a ref/tar image pair.
 * If --line y1,y2 is given, POIs are placed along two horizontal lines.
 * Otherwise, a grid of POIs covers the entire image (or ROI if specified).
 * Output: CSV with columns x,y,u,v,zncc per POI.
 */

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <omp.h>

#include "opencorr.h"

using namespace opencorr;
using namespace std;

int main(int argc, char* argv[])
{
    if (argc < 6) {
        cerr << "Usage: opencorr_cli <ref_image> <tar_image> <subset_radius> <grid_space> <output_csv>" << endl;
        cerr << "       [--line y1,y2] [--threads N]" << endl;
        cerr << endl;
        cerr << "  subset_radius: half of (subset_size - 1), e.g. 8 for subset_size=17" << endl;
        cerr << "  grid_space: step size in pixels between POIs" << endl;
        return 1;
    }

    string ref_path = argv[1];
    string tar_path = argv[2];
    int subset_radius = atoi(argv[3]);
    int grid_space = atoi(argv[4]);
    string output_path = argv[5];

    // Parse optional arguments
    int num_threads = omp_get_num_procs() - 1;
    if (num_threads < 1) num_threads = 1;
    vector<int> line_ys;

    for (int i = 6; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--line" && i + 1 < argc) {
            string val = argv[++i];
            stringstream ss(val);
            string token;
            while (getline(ss, token, ',')) {
                line_ys.push_back(atoi(token.c_str()));
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        }
    }

    // Load images
    Image2D ref_img(ref_path);
    Image2D tar_img(tar_path);
    int img_w = ref_img.width;
    int img_h = ref_img.height;

    cerr << "Image size: " << img_w << " x " << img_h << endl;
    cerr << "Subset radius: " << subset_radius << ", grid space: " << grid_space << endl;
    cerr << "Threads: " << num_threads << endl;

    omp_set_num_threads(num_threads);

    // Generate POIs
    vector<POI2D> poi_queue;
    int margin = subset_radius + 1;

    if (!line_ys.empty()) {
        // Place POIs along specified horizontal lines
        for (int y : line_ys) {
            if (y < margin || y >= img_h - margin) {
                cerr << "Warning: line y=" << y << " too close to edge, skipping" << endl;
                continue;
            }
            for (int x = margin; x < img_w - margin; x += grid_space) {
                Point2D pt((float)x, (float)y);
                POI2D poi(pt);
                poi_queue.push_back(poi);
            }
            cerr << "Line y=" << y << ": added POIs from x=" << margin << " to x=" << (img_w - margin - 1) << endl;
        }
    } else {
        // Grid over entire image
        for (int y = margin; y < img_h - margin; y += grid_space) {
            for (int x = margin; x < img_w - margin; x += grid_space) {
                Point2D pt((float)x, (float)y);
                POI2D poi(pt);
                poi_queue.push_back(poi);
            }
        }
    }

    cerr << "Total POIs: " << poi_queue.size() << endl;

    if (poi_queue.empty()) {
        cerr << "Error: no POIs generated" << endl;
        return 1;
    }

    // FFTCC for initial guess
    double t0 = omp_get_wtime();
    FFTCC2D* fftcc = new FFTCC2D(subset_radius, subset_radius, num_threads);
    fftcc->setImages(ref_img, tar_img);
    fftcc->compute(poi_queue);
    double t1 = omp_get_wtime();
    cerr << "FFTCC: " << (t1 - t0) << " sec" << endl;

    // ICGN1 for sub-pixel refinement
    int max_iteration = 10;
    float max_deformation_norm = 0.001f;
    ICGN2D1* icgn = new ICGN2D1(subset_radius, subset_radius, max_deformation_norm, max_iteration, num_threads);
    icgn->setImages(ref_img, tar_img);
    icgn->prepare();
    icgn->compute(poi_queue);
    double t2 = omp_get_wtime();
    cerr << "ICGN1: " << (t2 - t1) << " sec" << endl;

    // Output results as CSV
    ofstream out(output_path);
    if (!out.is_open()) {
        cerr << "Error: cannot open output file " << output_path << endl;
        return 1;
    }

    out << "x,y,u,v,zncc" << endl;
    for (const auto& poi : poi_queue) {
        out << poi.x << "," << poi.y << ","
            << poi.deformation.u << "," << poi.deformation.v << ","
            << poi.result.zncc << endl;
    }
    out.close();

    cerr << "Results saved to " << output_path << endl;
    cerr << "Total time: " << (t2 - t0) << " sec" << endl;

    delete fftcc;
    delete icgn;

    return 0;
}
