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
  offsets_(compartment_.get_offsets()),
  mols_(species_.get_mols()),
  blocks_(compartment_.get_model().get_blocks()),
  species_id_(species_.get_id()),
  vac_id_(species_.get_vac_id()),
  null_id_(species_.get_model().get_null_id()) {
}

void Diffuser::initialize() {
  Model& model(species_.get_model());
  stride_ = model.get_stride();
  id_stride_ = species_id_*stride_;
  is_reactive_.resize(model.get_species().size(), false);
  reactions_.resize(model.get_species().size(), NULL);
  //substrate_mols_.resize(model.get_species().size(), NULL);
  //product_mols_.resize(model.get_species().size(), NULL);
  //reacteds_.resize(mols_.size()+1, 0);
  const uint3& dimensions(compartment_.get_lattice_dimensions());
  num_voxels_ = dimensions.x*dimensions.y*dimensions.z;

  /*
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
  */
}

double Diffuser::get_D() const {
  return D_;
}

__device__
unsigned get_tar(
    const unsigned vdx,
    const unsigned nrand) {
  const bool odd_col((vdx%NUM_COLROW/NUM_ROW)&1);
  const bool odd_lay((vdx/NUM_COLROW)&1);
  switch(nrand)
    {
    case 1:
      return vdx+1;
    case 2:
      return vdx+(odd_col^odd_lay)-NUM_ROW-1 ;
    case 3:
      return vdx+(odd_col^odd_lay)-NUM_ROW;
    case 4:
      return vdx+(odd_col^odd_lay)+NUM_ROW-1;
    case 5:
      return vdx+(odd_col^odd_lay)+NUM_ROW;
    case 6:
      return vdx+NUM_ROW*(odd_lay-NUM_COL-1)-(odd_col&odd_lay);
    case 7:
      return vdx+!odd_col*(odd_lay-!odd_lay)-NUM_COLROW;
    case 8:
      return vdx+NUM_ROW*(odd_lay-NUM_COL)+(odd_col&!odd_lay);
    case 9:
      return vdx+NUM_ROW*(NUM_COL-!odd_lay)-(odd_col&odd_lay);
    case 10:
      return vdx+NUM_COLROW+!odd_col*(odd_lay-!odd_lay);
    case 11:
      return vdx+NUM_ROW*(NUM_COL+odd_lay)+(odd_col&!odd_lay);
    }
  return vdx-1;
}


// Bug fixed access: 12.5 BUPS
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  //unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  //const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  const unsigned block_jobs(mol_size_/gridDim.x);
  unsigned index(blockIdx.x*block_jobs);
  //unsigned end_index((blockIdx.x+1)*838860 + (threadIdx.x+1)*3276);
  unsigned end_index(index+block_jobs);
  while(index < end_index) {
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    index += blockDim.x;
    rand16 = (uint16_t)(rand32 >> 16);
    rand = ((uint32_t)rand16*12) >> 16;
    val = get_tar(mols_[index], rand);
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    index += blockDim.x;
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<64, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}

/*
// Bug fixed access: 12.5 BUPS
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  //unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  //const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  const unsigned block_jobs(mol_size_/gridDim.x);
  unsigned index(blockIdx.x*block_jobs);
  //unsigned end_index((blockIdx.x+1)*838860 + (threadIdx.x+1)*3276);
  unsigned end_index(index+block_jobs);
  while(index < end_index) {
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    index += blockDim.x;
    rand16 = (uint16_t)(rand32 >> 16);
    rand = ((uint32_t)rand16*12) >> 16;
    val = get_tar(mols_[index], rand);
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    index += blockDim.x;
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<64, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* Aligned mols_ access: 13.19 BUPS
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  //unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  //const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  const unsigned block_jobs(mol_size_/gridDim.x);
  unsigned index(blockIdx.x*block_jobs);
  //unsigned end_index((blockIdx.x+1)*838860 + (threadIdx.x+1)*3276);
  unsigned end_index(index+block_jobs);
  while(index < end_index) {
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    //index += blockDim.x;
    //if(index < end_index) {
      rand16 = (uint16_t)(rand32 >> 16);
      rand = ((uint32_t)rand16*12) >> 16;
      val = get_tar(mols_[index], rand);
      if(val < num_voxels_) {
        mols_[index] = val;
      }
      //Do nothing, stay at original position
      index += blockDim.x*2;
    //}
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<64, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* shared states: 13.18 BUPS
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  //unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  //const unsigned total_threads(blockDim.x*gridDim.x);
  __shared__ curandState local_state[256];
  local_state[threadIdx.x] = curand_states[blockIdx.x][threadIdx.x];
  const unsigned block_jobs(mol_size_/gridDim.x);
  unsigned index(blockIdx.x*block_jobs);
  //unsigned end_index((blockIdx.x+1)*838860 + (threadIdx.x+1)*3276);
  unsigned end_index(index+block_jobs);
  while(index < end_index) {
    const uint32_t rand32(curand(&local_state[threadIdx.x]));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    //index += blockDim.x;
    //if(index < end_index) {
      rand16 = (uint16_t)(rand32 >> 16);
      rand = ((uint32_t)rand16*12) >> 16;
      val = get_tar(mols_[index], rand);
      if(val < num_voxels_) {
        mols_[index] = val;
      }
      //Do nothing, stay at original position
      index += blockDim.x*2;
    //}
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state[threadIdx.x];
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<64, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* Reverted uint32_t for mol coord: 12.5 BUPS
__device__
unsigned get_tar(
    const unsigned vdx,
    const unsigned nrand) {
  const bool odd_col((vdx%NUM_COLROW/NUM_ROW)&1);
  const bool odd_lay((vdx/NUM_COLROW)&1);
  switch(nrand)
    {
    case 1:
      return vdx+1;
    case 2:
      return vdx+(odd_col^odd_lay)-NUM_ROW-1 ;
    case 3:
      return vdx+(odd_col^odd_lay)-NUM_ROW;
    case 4:
      return vdx+(odd_col^odd_lay)+NUM_ROW-1;
    case 5:
      return vdx+(odd_col^odd_lay)+NUM_ROW;
    case 6:
      return vdx+NUM_ROW*(odd_lay-NUM_COL-1)-(odd_col&odd_lay);
    case 7:
      return vdx+!odd_col*(odd_lay-!odd_lay)-NUM_COLROW;
    case 8:
      return vdx+NUM_ROW*(odd_lay-NUM_COL)+(odd_col&!odd_lay);
    case 9:
      return vdx+NUM_ROW*(NUM_COL-!odd_lay)-(odd_col&odd_lay);
    case 10:
      return vdx+NUM_COLROW+!odd_col*(odd_lay-!odd_lay);
    case 11:
      return vdx+NUM_ROW*(NUM_COL+odd_lay)+(odd_col&!odd_lay);
    }
  return vdx-1;
}


__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  while(index < mol_size_) {
    mols_[index] = curand(&local_state);
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    index += total_threads;
    if(index < mol_size_) {
      rand16 = (uint16_t)(rand32 >> 16);
      rand = ((uint32_t)rand16*12) >> 16;
      mol2_t val(get_tar(mols_[index], rand));
      if(val < num_voxels_) {
        mols_[index] = val;
      }
      //Do nothing, stay at original position
      index += total_threads;
    }
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/


/* kernel<<<3, 5>>>()
   <-block0-><-block1-><-block2->
   |0|1|2|3|4|0|1|2|3|4|0|1|2|3|4|
   gridDim = number of blocks in a grid = 3
   blockDim = number of threads per block = 5
   blockIdx = index of the block = [0,1,2]
   threadIdx = index of the thread in a block = [0,1,2,3,4]
   980 GTX: multiProcessorCount = 16
*/


/* With uint3 as coord of mols: 2.9 BUPS

__device__
void set_tar(
    umol_t& val,
    const unsigned nrand) {
  const umol_t vdx(val);
  switch(nrand) {
    case 0:
        val.y += -1;
      break;
      //return vdx-1;
    case 1:
        val.y += 1;
      break;
      //return vdx+1;
    case 2:
      val.x += -1;
      val.y += ((vdx.x&1)^(vdx.z&1))-1;
      break;
      //return vdx+(odd_col^odd_lay)-NUM_ROW-1 ;
    case 3:
      val.x += -1;
      val.y += ((vdx.x&1)^(vdx.z&1));
      break;
      //return vdx+(odd_col^odd_lay)-NUM_ROW;
    case 4:
      val.x += 1;
      val.y += ((vdx.x&1)^(vdx.z&1))-1;
      break;
      //return vdx+(odd_col^odd_lay)+NUM_ROW-1;
    case 5:
      val.x += 1;
      val.y += ((vdx.x&1)^(vdx.z&1));
      break;
      //return vdx+(odd_col^odd_lay)+NUM_ROW;
    case 6:
      val.x += (vdx.z&1)-1;
      val.y += -((vdx.x&1)&(vdx.z&1));
      val.z += -1;
      break;
      //return vdx+NUM_ROW*(odd_lay-NUM_COL-1)-(odd_col&odd_lay);
    case 7:
      val.y += !(vdx.x&1)*((vdx.z&1)-!(vdx.z&1));
      val.z += -1;
      break;
      //return vdx+!odd_col*(odd_lay-!odd_lay)-NUM_COLROW;
    case 8:
      val.x += vdx.z&1;
      val.y += ((vdx.x&1)&!(vdx.z&1));
      val.z += -1;
      break;
      //return vdx+NUM_ROW*(odd_lay-NUM_COL)+(odd_col&!odd_lay);
    case 9:
      val.x += -!(vdx.z&1);
      val.y += -((vdx.x&1)&(vdx.z&1));
      val.z += 1;
      break;
      //return vdx+NUM_ROW*(NUM_COL-!odd_lay)-(odd_col&odd_lay);
    case 10:
      val.y += !(vdx.x&1)*((vdx.z&1)-!(vdx.z&1));
      val.z += 1;
      break;
      //return vdx+NUM_COLROW+!odd_col*(odd_lay-!odd_lay);
    case 11:
      val.x += vdx.z&1;
      val.y += ((vdx.x&1)&!(vdx.z&1));
      val.z += 1;
      break;
      //return vdx+NUM_ROW*(NUM_COL+odd_lay)+(odd_col&!odd_lay);
    }
}

__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const uimol_t num_voxels_,
    umol_t* mols_) {
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  while(index < mol_size_) {
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    set_tar(mols_[index], rand);
    index += total_threads;
    if(index < mol_size_) {
      rand16 = (uint16_t)(rand32 >> 16);
      rand = ((uint32_t)rand16*12) >> 16;
      set_tar(mols_[index], rand);
      index += total_threads;
    }
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* reduced global memory: 12.54 BUPS vs max 41.3 BUPS for rand generation
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  while(index < mol_size_) {
    mols_[index] = curand(&local_state);
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    index += total_threads;
    if(index < mol_size_) {
      rand16 = (uint16_t)(rand32 >> 16);
      rand = ((uint32_t)rand16*12) >> 16;
      mol2_t val(get_tar(mols_[index], rand));
      if(val < num_voxels_) {
        mols_[index] = val;
      }
      //Do nothing, stay at original position
      index += total_threads;
    }
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* splitting uint32_t rand into two uint16_t: 22.4 s 
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  while(index < mol_size_) {
    const uint32_t rand32(curand(&local_state));
    uint16_t rand16((uint16_t)(rand32 & 0x0000FFFFuL));
    uint32_t rand(((uint32_t)rand16*12) >> 16);
    mol2_t val(get_tar(mols_[index], rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    index += total_threads;
    if(index < mol_size_) {
      rand16 = (uint16_t)(rand32 >> 16);
      rand = ((uint32_t)rand16*12) >> 16;
      mol2_t val(get_tar(mols_[index], rand));
      if(val < num_voxels_) {
        mols_[index] = val;
      }
      //Do nothing, stay at original position
      index += total_threads;
    }
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* With persistent local curand_states: 2.3 s or 22.8 s
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states[blockIdx.x][threadIdx.x];
  while(index < mol_size_) {
    const umol_t vdx(mols_[index]);
    float ranf(curand_uniform(&local_state)*11.999999);
    const unsigned rand((unsigned)truncf(ranf));
    mol2_t val(get_tar(vdx, rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    index += total_threads;
  }
  curand_states[blockIdx.x][threadIdx.x] = local_state;
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* Without generated randoms: 2.6 s
__global__
void concurrent_walk(
    const unsigned mol_size_,
    const voxel_t stride_,
    const voxel_t id_stride_,
    const voxel_t vac_id_,
    const voxel_t null_id_,
    const umol_t num_voxels_,
    curandState* curand_states_,
    umol_t* mols_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  curandState local_state = curand_states_[index];
  while(index < mol_size_) {
    const umol_t vdx(mols_[index]);
    float ranf(curand_uniform(&local_state)*11.999999);
    const unsigned rand((unsigned)truncf(ranf));
    mol2_t val(get_tar(vdx, rand));
    if(val < num_voxels_) {
      mols_[index] = val;
    }
    //Do nothing, stay at original position
    index += total_threads;
  }
  curand_states_[index] = local_state;
  //__syncthreads();
}

void Diffuser::walk() {
  const size_t size(mols_.size());
  concurrent_walk<<<blocks_, 256>>>(
      size,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      compartment_.get_model().get_curand_states(),
      thrust::raw_pointer_cast(&mols_[0]));
}
*/

/* Without offsets: 3.0 s
__device__
unsigned get_tar(
    const unsigned vdx,
    const unsigned nrand) {
  const bool odd_col((vdx%NUM_COLROW/NUM_ROW)&1);
  const bool odd_lay((vdx/NUM_COLROW)&1);
  switch(nrand)
    {
    case 1:
      return vdx+1;
    case 2:
      return vdx+(odd_col^odd_lay)-NUM_ROW-1 ;
    case 3:
      return vdx+(odd_col^odd_lay)-NUM_ROW;
    case 4:
      return vdx+(odd_col^odd_lay)+NUM_ROW-1;
    case 5:
      return vdx+(odd_col^odd_lay)+NUM_ROW;
    case 6:
      return vdx+NUM_ROW*(odd_lay-NUM_COL-1)-(odd_col&odd_lay);
    case 7:
      return vdx+!odd_col*(odd_lay-!odd_lay)-NUM_COLROW;
    case 8:
      return vdx+NUM_ROW*(odd_lay-NUM_COL)+(odd_col&!odd_lay);
    case 9:
      return vdx+NUM_ROW*(NUM_COL-!odd_lay)-(odd_col&odd_lay);
    case 10:
      return vdx+NUM_COLROW+!odd_col*(odd_lay-!odd_lay);
    case 11:
      return vdx+NUM_ROW*(NUM_COL+odd_lay)+(odd_col&!odd_lay);
    }
  return vdx-1;
}

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
    const float* randoms_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  while(index < mol_size_) {
    const umol_t vdx(mols_[index]);
    float ranf(randoms_[index]*11.999999);
    const unsigned rand((unsigned)truncf(ranf));
    mol2_t val(get_tar(vdx, rand));
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
  if(randoms_counter_ > compartment_.get_model().get_randoms_size()-size) {
    compartment_.get_model().generate_randoms();
    randoms_counter_ = 0;
  }
  concurrent_walk<<<blocks_, 512>>>(
      size,
      seed_,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]),
      thrust::raw_pointer_cast(&randoms_[randoms_counter_]));
  randoms_counter_ += size;
  //barrier cudaDeviceSynchronize() is not needed here since all work will be
  //queued in the stream sequentially by the CPU to be executed by GPU.
  //kernel1<<<X,Y>>>(...); // kernel start execution, CPU continues to next
                           // statement
  //kernel2<<<X,Y>>>(...); // kernel is placed in queue and will start after
                           // kernel1 finishes, CPU continues to next statement
  //cudaMemcpy(...); // CPU blocks until ememory is copied, memory copy starts
                     // only after kernel2 finishes

}
*/

/* curand random generation: 4.1 s
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
    const mol_t* offsets_,
    const float* randoms_) {
  //index is the unique global thread id (size: total_threads)
  unsigned index(blockIdx.x*blockDim.x + threadIdx.x);
  const unsigned total_threads(blockDim.x*gridDim.x);
  while(index < mol_size_) {
    const umol_t vdx(mols_[index]);
    float ranf(randoms_[index]*11.999999);
    const unsigned rand((unsigned)truncf(ranf));
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
  if(randoms_counter_ > compartment_.get_model().get_randoms_size()-size) {
    compartment_.get_model().generate_randoms();
    randoms_counter_ = 0;
  }
  concurrent_walk<<<blocks_, 512>>>(
      size,
      seed_,
      stride_,
      id_stride_,
      vac_id_,
      null_id_,
      num_voxels_,
      thrust::raw_pointer_cast(&mols_[0]),
      thrust::raw_pointer_cast(&offsets_[0]),
      thrust::raw_pointer_cast(&randoms_[randoms_counter_]));
  randoms_counter_ += size;
  //barrier cudaDeviceSynchronize() is not needed here since all work will be
  //queued in the stream sequentially by the CPU to be executed by GPU.
  //kernel1<<<X,Y>>>(...); // kernel start execution, CPU continues to next
                           // statement
  //kernel2<<<X,Y>>>(...); // kernel is placed in queue and will start after
                           // kernel1 finishes, CPU continues to next statement
  //cudaMemcpy(...); // CPU blocks until ememory is copied, memory copy starts
                     // only after kernel2 finishes

}
*/

