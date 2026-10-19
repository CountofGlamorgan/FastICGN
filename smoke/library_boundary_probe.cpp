/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Public FFTCC entry guards and cache miss lifecycle; runs without external data.
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>
#ifdef OPENCORR_GPU_MVP_USE_CUDA
#include "fftcc2d_cuda.h"
#endif
#ifdef OPENCORR_HAS_FFTW
#include "oc_fftcc.h"
#endif
using namespace opencorr;
namespace {
int failures = 0;
void check(const char* label, bool value) { failures += !value; std::cout << (value ? "PASS " : "FAIL ") << label << '\n'; }
template<class F> void rejects(const char* label, F operation) {
  bool ok = false; try { operation(); } catch (const std::invalid_argument&) { ok = true; }
  check(label, ok);
}
bool same(const std::vector<POI2D>& a, const std::vector<POI2D>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i=0;i<a.size();++i)
    if (a[i].deformation.u != b[i].deformation.u || a[i].deformation.v != b[i].deformation.v
        || a[i].result.zncc != b[i].result.zncc) return false;
  return true;
}
template<class Solver, class Factory> void exercise(Factory fresh) {
  Image2D ref(192,128), tar(192,128);
  unsigned state=12345;
  for(int y=0;y<128;++y) for(int x=0;x<192;++x) {
    state=1664525*state+1013904223;ref.eg_mat(y,x)=float(state>>24);
  }
  for(int y=0;y<128;++y) for(int x=0;x<192;++x)
    tar.eg_mat(y,x)=(x>=2 && y+1<128)?ref.eg_mat(y+1,x-2):127.f;
  std::vector<POI2D> layout;
  for(int y=32;y<96;y+=16) for(int x=32;x<160;x+=16) layout.emplace_back(x,y);
  auto solver=fresh();solver->setImages(ref,tar);solver->prepareRef(layout);
  check("prepared-original-layout",solver->hasPreparedReference(ref));
  solver->setRefImage(ref);
  auto different=layout;different[0].x+=1;
  auto expected=different;auto uncached=fresh();uncached->setImages(ref,tar);uncached->compute(expected);
  solver->compute(different);
  check("mismatched-queue-does-not-rebuild-old-cache",!solver->hasPreparedReference(ref));
  check("mismatched-queue-matches-fresh",same(different,expected));
  auto matching=layout;solver->compute(matching);
  check("matching-queue-rebuilds-retained-layout",solver->hasPreparedReference(ref));
  int correct=0;
  for(const auto& p:matching) correct += p.deformation.u==2 && p.deformation.v==-1 && p.result.zncc>.5;
  check("square-translation-truth",correct==int(layout.size()));
  solver->setRefImage(ref);
  auto indexed=layout;
  rejects("indexed-layout-mismatch-rejected",[&]{solver->compute(indexed,std::vector<size_t>(layout.size(),0));});
}
template<class Factory> void cacheTest(Factory fresh) {
  // Avoid catching and hiding errors inside the tested state transitions.
  using Solver=typename decltype(fresh())::element_type;
  exercise<Solver>(fresh);
}
#ifdef OPENCORR_GPU_MVP_USE_CUDA
// review-2026-10-03 (fftcc F3): the target extraction is bounds-checked against
// the reference size, so a smaller target used to be read out of bounds and
// the queue came back filled with garbage. A size mismatch must throw, name
// both sizes and leave the queue and the solver usable.
void gpuImageSizeMismatch() {
  auto noise=[](Image2D& image,unsigned state) {
    for(int y=0;y<image.height;++y) for(int x=0;x<image.width;++x) {
      state=1664525*state+1013904223;image.eg_mat(y,x)=float(state>>24);
    }
  };
  Image2D ref(192,128), tar(192,128), smaller(96,64), larger(256,160);
  noise(ref,12345);noise(smaller,23456);noise(larger,34567);
  for(int y=0;y<128;++y) for(int x=0;x<192;++x)
    tar.eg_mat(y,x)=(x>=2 && y+1<128)?ref.eg_mat(y+1,x-2):127.f;
  std::vector<POI2D> layout;
  for(int y=32;y<96;y+=16) for(int x=32;x<160;x+=16) layout.emplace_back(x,y);
  std::vector<size_t> identity(layout.size());
  for(size_t i=0;i<identity.size();++i) identity[i]=i;
  FFTCC2DCuda solver(7,7);solver.setImages(ref,tar);solver.prepareRef(layout);
  auto expected=layout;solver.compute(expected);
  const float sentinel=-7.f;
  auto untouched=[&](const std::vector<POI2D>& queue) {
    for(size_t i=0;i<queue.size();++i)
      if(queue[i].result.zncc!=sentinel || queue[i].deformation.u!=0.f || queue[i].deformation.v!=0.f
         || queue[i].x!=layout[i].x || queue[i].y!=layout[i].y) return false;
    return true;
  };
  for(const auto& other:std::vector<std::pair<const Image2D*,std::string>>{{&smaller,"96x64"},{&larger,"256x160"}}) {
    solver.setTarImage(*other.first);
    for(bool indexed:{false,true}) {
      auto queue=layout;for(auto& p:queue) p.result.zncc=sentinel;
      bool rejected=false;std::string message;
      try { if(indexed) solver.compute(queue,identity); else solver.compute(queue); }
      catch(const std::invalid_argument& ex) { rejected=true;message=ex.what(); }
      const std::string label="GPU-target-"+other.second+"-indexed-"+std::to_string(indexed);
      check((label+"-rejected").c_str(),rejected);
      check((label+"-message-names-both-sizes").c_str(),
            message.find("192x128")!=std::string::npos && message.find(other.second)!=std::string::npos);
      check((label+"-queue-untouched").c_str(),untouched(queue));
    }
  }
  solver.setTarImage(tar);
  auto again=layout;solver.compute(again);
  check("GPU-matching-size-after-rejection-matches-first-run",same(again,expected));
}
#endif
}
int main() {
  omp_set_num_threads(2);
  try {
#ifdef OPENCORR_GPU_MVP_USE_CUDA
    for(auto r:std::vector<std::pair<int,int>>{{7,15},{15,7},{0,0},{-1,-1},{23171,23171}})
      rejects("GPU-invalid-or-rectangular-radius",[&]{ FFTCC2DCuda solver(r.first,r.second); });
    cacheTest([]{return std::make_unique<FFTCC2DCuda>(7,7);});
    gpuImageSizeMismatch();
#endif
#ifdef OPENCORR_HAS_FFTW
    for(auto r:std::vector<std::pair<int,int>>{{7,15},{15,7},{0,0},{-1,-1},{23171,23171}})
      rejects("CPU-invalid-or-rectangular-radius",[&]{ FFTCC2D solver(r.first,r.second,2); });
    rejects("CPU-invalid-thread-count",[]{FFTCC2D solver(7,7,0);});
    cacheTest([]{return std::make_unique<FFTCC2D>(7,7,2);});
    FFTCC2D solver(7,7,2);DIC& base=solver;base.setSubset(15,7);
    rejects("CPU-base-subset-mutation",[&]{solver.prepareRef();});
    base.setSubset(15,15);
    rejects("CPU-square-subset-resize",[&]{solver.prepareRef();});
#endif
  } catch(const std::exception& ex) {std::cerr<<ex.what()<<'\n';return 2;}
  catch(const std::string& ex) {std::cerr<<ex<<'\n';return 2;}
  std::cout<<"library boundary failures="<<failures<<'\n';return failures?1:0;
}
