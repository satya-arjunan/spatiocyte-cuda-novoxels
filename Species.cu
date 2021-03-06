//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of the Spatiocyte package
//
//        Copyright (C) 2006-2009 Keio University
//        Copyright (C) 2010-2013 RIKEN
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

#include <iostream>
#include <algorithm>
#include <string>
#include <time.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <Species.hpp>
#include <Compartment.hpp>
#include <Reaction.hpp>
#include <Model.hpp>

Species::Species(const std::string name, const unsigned nmols, const double D,
    Model& model, Compartment& compartment, Species& vacant,
    const bool is_structure_species):
  compartment_(compartment),
  model_(model),
  vacant_(vacant),
  name_(get_init_name(name)),
  init_nmols_(nmols),
  is_structure_species_(is_structure_species),
  id_(model.push_species(*this)),
  vac_id_(vacant_.get_id()),
  diffuser_(D, *this) {
}

void Species::initialize() {
  if(!is_structure_species_) {
    populate();
  }
  if(mols_.size() < 80000) {
    mols_.reserve(10000);
  }
  diffuser_.initialize();
}

void Species::push_reaction(Reaction& reaction) {
  if(std::find(reactions_.begin(), reactions_.end(), &reaction) ==
      reactions_.end()) { 
    reactions_.push_back(&reaction);
  }
}

std::vector<Reaction*>& Species::get_reactions() {
  return reactions_;
}

__host__ __device__
uimol_t umol_to_uimol(const umol_t& umol, const uint3& dimensions) {
  return umol;
  //return umol.y + umol.x*dimensions.y + umol.z*dimensions.x*dimensions.y;
}

__host__ __device__
umol_t uimol_to_umol(const uimol_t& uimol, const uint3& dimensions) {
  return uimol;
  /*
  const unsigned xy(dimensions.x*dimensions.y);
  //(cols, row, lay)
  return make_uint3(uimol%xy/dimensions.y, uimol%xy%dimensions.y, uimol/xy);
  */
}



//overlapping molecule population
struct populate_lattice {
  __host__ __device__ populate_lattice(
      const unsigned seed,
      const uimol_t mol_size,
      const voxel_t vac_id,
      const voxel_t stride_id,
      const uint3 dimensions):
    seed_(seed),
    mol_size_(mol_size),
    vac_id_(vac_id),
    stride_id_(stride_id),
    dimensions_(dimensions) {}
  __device__ umol_t operator()(const unsigned n) const {
    thrust::default_random_engine rng(seed_);
    rng.discard(n);
    thrust::uniform_int_distribution<unsigned> u(0, dimensions_.x*
        dimensions_.y*dimensions_.z);
    uimol_t rand(u(rng));
    return umol_t(uimol_to_umol(u(rng), dimensions_));
  }
  const unsigned seed_;
  const uimol_t mol_size_;
  const voxel_t vac_id_;
  const voxel_t stride_id_;
  const uint3 dimensions_;
};


void Species::populate() {
  mols_.resize(init_nmols_);
  cudaDeviceSynchronize();
  const Vector<unsigned>& dimensions();
  thrust::transform(thrust::device, 
      thrust::counting_iterator<unsigned>(0),
      thrust::counting_iterator<unsigned>(init_nmols_),
      mols_.begin(),
      populate_lattice(
        id_,
        init_nmols_,
        vac_id_,
        get_id()*model_.get_stride(),
        compartment_.get_lattice_dimensions()));
}

void Species::populate_in_lattice() {
  mols_.resize(host_mols_.size());
  thrust::copy(host_mols_.begin(), host_mols_.end(), mols_.begin());
  if(diffuser_.get_D()) {
    host_mols_.clear();
    std::vector<umol_t>().swap(host_mols_);
  } else {
    mols_.clear();
    thrust::device_vector<umol_t>().swap(mols_);
  }
}

void Species::push_host_mol(const umol_t vdx) {
  host_mols_.push_back(vdx);
}

bool Species::is_structure_species() const {
  return is_structure_species_;
}

bool Species::is_root_structure_species() const {
  return (this == &vacant_);
}

Model& Species::get_model() const {
  return model_;
}

voxel_t Species::get_id() const {
  return id_;
}

voxel_t Species::get_vac_id() const {
  return vac_id_;
}

Diffuser& Species::get_diffuser() {
  return diffuser_;
}

Compartment& Species::get_compartment() {
  return compartment_;
}

Species& Species::get_vacant() {
  return vacant_;
}

const std::string& Species::get_name() const {
  return name_;
}

const std::string Species::get_name_id() const {
  /*
  std::stringstream sid;
  sid << (unsigned)get_id();
  return std::string(get_name()+":"+sid.str());
  */
  return std::string(get_name()+":id:"+std::to_string(get_id())); // c++11
}

const std::string Species::get_init_name(const std::string name) const {
  if(is_root_structure_species()) {
    return std::string(compartment_.get_name()+"/"+name);
  }
  return std::string(vacant_.get_name()+"/"+name);
}

std::vector<umol_t>& Species::get_host_mols() {
  if(diffuser_.get_D()) {
    host_mols_.resize(mols_.size());
    thrust::copy(mols_.begin(), mols_.end(), host_mols_.begin());
  }
  return host_mols_;
}

thrust::device_vector<umol_t>& Species::get_mols() {
  return mols_;
}
