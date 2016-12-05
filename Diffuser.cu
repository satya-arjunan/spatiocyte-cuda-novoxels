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

#include <time.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/system/cuda/detail/bulk/bulk.hpp>
#include <curand_kernel.h>
#include <Diffuser.hpp>
#include <Compartment.hpp>
#include <Model.hpp>
#include <Reaction.hpp>
#include <random>

Diffuser::Diffuser(const double D, Species& species):
  D_(D),
  species_(species),
  compartment_(species_.get_compartment()),
  mols_(species_.get_mols()),
  offsets_(species_.get_compartment().get_offsets()),
  species_id_(species_.get_id()),
  vac_id_(species_.get_vac_id()),
  null_id_(species_.get_model().get_null_id()),
  seed_(0) {
}

void Diffuser::initialize() {
  Model& model(species_.get_model());
  stride_ = model.get_stride();
  id_stride_ = species_id_*stride_;
  is_reactive_.resize(model.get_species().size(), false);
  reactions_.resize(model.get_species().size(), NULL);
  substrate_mols_.resize(model.get_species().size(), NULL);
  product_mols_.resize(model.get_species().size(), NULL);
  reacteds_.resize(mols_.size()+1, 0);
  const Vector<unsigned>& dimensions(compartment_.get_lattice_dimensions());
  num_voxels_ = dimensions.x*dimensions.y*dimensions.z;

  std::vector<Reaction*>& reactions(species_.get_reactions());
  for(unsigned i(0); i != reactions.size(); ++i) {
    std::vector<Species*>& substrates(reactions[i]->get_substrates());
    for(unsigned j(0); j != substrates.size(); ++j) {
      voxel_t reactant_id(substrates[j]->get_id());
      if(reactant_id != species_id_) {
        reactions_[reactant_id] = reactions[i];
        is_reactive_[reactant_id] = true;
        substrate_mols_[reactant_id] = thrust::raw_pointer_cast(substrates[j]->get_mols().data());
        product_mols_[reactant_id] = thrust::raw_pointer_cast(reactions[i]->get_products()[0]->get_mols().data());
      } 
    } 
  } 
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  //better performance when the number of blocks is twice the number of 
  //multi processors (aka streams):
  blocks_ = prop.multiProcessorCount*2;
  std::cout << "number of blocks:" << blocks_ << " maj:" << prop.major << " min:" << prop.minor << std::endl;
  /*
  std::cout << "My name:" << species_.get_name_id() << std::endl;
  for(unsigned i(0); i != is_reactive_.size(); ++i) {
    std::cout << "\t" << is_reactive_[i] << " reactant name:" << model.get_species()[i]->get_name_id() << std::endl;
    std::cout << "\t" << (reactions_[i] != NULL) << std::endl;
  }
  */
}

double Diffuser::get_D() const {
  return D_;
}

/* kernel<<<3, 5>>>()
   <-block0-><-block1-><-block2->
   |0|1|2|3|4|0|1|2|3|4|0|1|2|3|4|
   gridDim = number of blocks in a grid = 3
   blockDim = number of threads per block = 5
   blockIdx = index of the block = [0,1,2]
   threadIdx = index of the thread in a block = [0,1,2,3,4]
   980 GTX: multiProcessorCount = 16
*/

__global__
void concurrent_walk(
    const unsigned mol_size_,
    const unsigned seed_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_,
    const mol_t* offsets_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  while(index < mol_size_) {
    const umol_t vdx(mols_[index]);
    thrust::default_random_engine rng;
    rng.discard(seed_+index);
    thrust::uniform_int_distribution<unsigned> u(0, 11);
    const unsigned rand(u(rng));
    const unsigned lay(vdx/NUM_COLROW);
    const unsigned col(vdx%NUM_COLROW/NUM_ROW);
    const bool odd_lay(lay&1);
    const bool odd_col(col&1);
    mol2_t val(mol2_t(vdx)+offsets_[rand+(24&(-odd_lay))+(12&(-odd_col))]);
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    index += total_threads;
  }
  //__syncthreads();
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 512>>>(
      size,
      seed_,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]),
      thrust::raw_pointer_cast(&offsets_[0]));
  //barrier cudaDeviceSynchronize() is not needed here since all work will be
  //queued in the stream sequentially by the CPU to be executed by GPU.
  //kernel1<<<X,Y>>>(...); // kernel start execution, CPU continues to next
                           // statement
  //kernel2<<<X,Y>>>(...); // kernel is placed in queue and will start after
                           // kernel1 finishes, CPU continues to next statement
  //cudaMemcpy(...); // CPU blocks until ememory is copied, memory copy starts
                     // only after kernel2 finishes
  seed_ += size;
}

