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


#ifndef __Diffuser_hpp
#define __Diffuser_hpp

#include <thrust/device_vector.h>
#include <Common.hpp>

class Diffuser
{ 
public: 
  Diffuser(const double, Species&);
  ~Diffuser() {}
  void initialize();
  void walk();
  double get_D() const;
private:
  void set_offsets();
  const double D_;
  Species& species_;
  Compartment& compartment_;
  thrust::device_vector<umol_t>& mols_;
  thrust::device_vector<mol_t>& offsets_;
  const voxel_t species_id_;
  const voxel_t vac_id_;
  const voxel_t null_id_;
  unsigned seed_;
  unsigned blocks_;
  voxel_t stride_;
  voxel_t id_stride_;
  umol_t num_voxels_;
  thrust::device_vector<bool> is_reactive_;
  thrust::device_vector<Reaction*> reactions_;
  thrust::device_vector<umol_t> reacteds_;
  thrust::device_vector<umol_t*> substrate_mols_;
  thrust::device_vector<umol_t*> product_mols_;
};

#endif /* __Diffuser_hpp */

