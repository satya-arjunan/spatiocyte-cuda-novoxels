//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of the Spatiocyte package
//
//        Copyright (C) 2006-2009 Keio University
//        Copyright (C) 2010-2014 RIKEN
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Spatiocyte is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Spatiocyte is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Spatiocyte -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Satya Arjunan <satya.arjunan@gmail.com>
//

#include <stdexcept>
#include <Spatiocyte.hpp>
#include <Model.hpp>
#include <math.h>

__device__ curandState* curand_states[64];

Model::Model():
  null_id_((voxel_t)(pow(2,sizeof(voxel_t)*8))),
  randoms_size_(536870912),
  randoms_counter_(0),
  compartment_("root", LENGTH_X, LENGTH_Y, LENGTH_Z, *this) {
} 

Model::~Model() {
  //curandDestroyGenerator(random_generator_);
}

void Model::initialize() {
  stride_ = null_id_/species_.size();
  //std::cout << "species size:" << species_.size() << " null_id:" << null_id_ << " stride:" << stride_ <<  std::endl;
  compartment_.initialize();
  for (unsigned i(0), n(species_.size()); i != n; ++i) {
      species_[i]->initialize();
    }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  //better performance when the number of blocks is twice the number of 
  //multi processors (aka streams):
  blocks_ = prop.multiProcessorCount*4;
  std::cout << "number blocks:" << blocks_ << std::endl;
  //initialize_randoms();
  initialize_random_generator();
}

//Setup the default XORWOW generator:
__global__
void setup_kernel2() {
  int id = threadIdx.x + blockIdx.x * 256;
  if(threadIdx.x == 0) {
    curand_states[blockIdx.x] = 
      (curandState*)malloc(blockDim.x*sizeof(curandState));
  }
  __syncthreads();
  curand_init(1234, id, 0, &curand_states[blockIdx.x][threadIdx.x]);
}
/*
__global__
void setup_kernel2() {
  int id = threadIdx.x + blockIdx.x * 256;
  if(threadIdx.x == 0) {
    curand_states[blockIdx.x] = 
      (curandState*)malloc(blockDim.x*sizeof(curandState));
  }
  __syncthreads();
  curand_init(1234, id, 0, &curand_states[blockIdx.x][threadIdx.x]);
}
*/

__global__
void setup_kernel(curandState *state) {
  int id = threadIdx.x + blockIdx.x * 256;
  /* Each thread gets same seed, a different sequence number, no offset */
  curand_init(1234, id, 0, &state[id]);
}


void Model::initialize_randoms() {
  randoms_.resize(randoms_size_);
  //Ordered from fastest to slowest:
  curandCreateGenerator(&random_generator_, CURAND_RNG_PSEUDO_XORWOW);
  //curandCreateGenerator(&random_generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  //curandCreateGenerator(&random_generator_, CURAND_RNG_PSEUDO_MT19937);
  //curandCreateGenerator(&random_generator_, CURAND_RNG_PSEUDO_MTGP32);
  //curandCreateGenerator(&random_generator_, CURAND_RNG_PSEUDO_MRG32K3A);
  curandSetPseudoRandomGeneratorSeed(random_generator_, 1234ULL);
  generate_randoms();
}

/*
void Model::initialize_random_generator() {
  setup_kernel<<<blocks_, 256>>>();
}
*/

void Model::initialize_random_generator() {
  cudaMalloc((void **)&curand_states_, blocks_*256 * sizeof(curandState));
  setup_kernel<<<blocks_, 256>>>(curand_states_);
}

void Model::generate_randoms() {
  curandGenerateUniform(random_generator_,
      thrust::raw_pointer_cast(&randoms_[0]), randoms_size_);
}

curandState* Model::get_curand_states() {
  return curand_states_;
}

thrust::device_vector<float>& Model::get_randoms() {
  return randoms_;
}

unsigned& Model::get_randoms_counter() {
  return randoms_counter_;
}

unsigned& Model::get_blocks() {
  return blocks_;
}

unsigned Model::get_randoms_size() const {
  return randoms_size_;
}

voxel_t Model::get_null_id() const {
  return null_id_;
}

voxel_t Model::get_stride() const {
  return stride_;
}

unsigned Model::run(const double interval) {
  const unsigned steps(interval/4.16667e-6);
  for (unsigned i(0); i != steps; ++i) {
      stepper_.step();
    }
  return steps;
}

unsigned Model::push_species(Species& species) {
  species_.push_back(&species);
  return species_.size()-1;
}

Compartment& Model::get_compartment() {
  return compartment_;
}

Stepper& Model::get_stepper() {
  return stepper_;
}

std::vector<Species*>& Model::get_species() {
  return species_;
}

