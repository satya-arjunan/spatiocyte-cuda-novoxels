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

Model::Model():
  null_id_((voxel_t)(pow(2,sizeof(voxel_t)*8))),
  compartment_("root", LENGTH_X, LENGTH_Y, LENGTH_Z, *this) {
} 

void Model::initialize() {
  stride_ = null_id_/species_.size();
  //std::cout << "species size:" << species_.size() << " null_id:" << null_id_ << " stride:" << stride_ <<  std::endl;
  compartment_.initialize();
  for (unsigned i(0), n(species_.size()); i != n; ++i) {
      species_[i]->initialize();
    }
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

