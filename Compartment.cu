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

#include <math.h>
#include <Compartment.hpp>
#include <Species.hpp>
#include <Model.hpp>

Compartment::Compartment(std::string name, const double len_x,
    const double len_y, const double len_z, Model& model):
  name_(name),
  model_(model),
  lattice_dimensions_(NUM_COL, NUM_ROW, NUM_LAY),
  dimensions_(NUM_COL*HCP_X, 
              NUM_ROW*2*VOXEL_RADIUS,
              NUM_LAY*HCP_Z),
  volume_species_("volume", 0, 0, model, *this, volume_species_, true),
  surface_species_("surface", 0, 0, model, *this, volume_species_, true) {
}

void Compartment::initialize() {
  std::cout << "Volume:" << dimensions_.x*dimensions_.y*dimensions_.z <<
    " m^3" << std::endl;
  double num_voxel(double(NUM_COL)*NUM_ROW*NUM_LAY);
  double max_umol_t(pow(2,sizeof(umol_t)*8));
  if(num_voxel > max_umol_t)
    {
      std::cout << "Number of voxels:" << num_voxel <<
        " exceeds max value of umol_t:" << max_umol_t << std::endl;
      std::cout << "Reduce lattice lengths or change umol_t type in " <<
        "Common.hpp. Exiting now..." << std::endl;
      exit(0);
    }
  set_volume_structure();
  set_surface_structure();
}

const Vector<unsigned>& Compartment::get_lattice_dimensions() const {
  return lattice_dimensions_;
}

const Vector<double>& Compartment::get_dimensions() const {
  return dimensions_;
}

Vector<double> Compartment::get_center() const {
  return dimensions_/2;
}

Species& Compartment::get_surface_species() {
  return surface_species_;
}

Species& Compartment::get_volume_species() {
  return volume_species_;
}

Model& Compartment::get_model() {
  return model_;
}

const std::string& Compartment::get_name() const {
  return name_;
}

void Compartment::set_volume_structure() {
  /*
  for(umol_t i(0); i != get_lattice().get_num_voxel(); ++i) {
    get_volume_species().push_host_mol(i);
  }
  */
}

void Compartment::set_surface_structure() {
  for (umol_t i(0); i != NUM_COLROW; ++i) {
    get_surface_species().push_host_mol(i);
    get_surface_species().push_host_mol(NUM_VOXEL-1-i);
  }
  for (umol_t i(1); i != NUM_LAY-1; ++i) {
    //layer_row yz-plane
    for (umol_t j(0); j != NUM_ROW; ++j) {
      get_surface_species().push_host_mol(i*NUM_COLROW+j);
      get_surface_species().push_host_mol(i*NUM_COLROW+j+NUM_ROW*
                                         (NUM_COL-1));
    }
    //layer_col xz-plane
    for (umol_t j(1); j != NUM_COL-1; ++j) {
      get_surface_species().push_host_mol(i*NUM_COLROW+j*NUM_ROW);
      get_surface_species().push_host_mol(i*NUM_COLROW+j*NUM_ROW+
                                         NUM_ROW-1);
    }
  }
  get_surface_species().populate_in_lattice();
}


