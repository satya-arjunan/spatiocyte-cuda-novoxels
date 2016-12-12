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
  offsets_(ADJS*4),
  lattice_dimensions_(make_uint3(NUM_COL, NUM_ROW, NUM_LAY)),
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
  double max_uimol_t(pow(2,sizeof(uimol_t)*8));
  if(num_voxel > max_uimol_t)
    {
      std::cout << "Number of voxels:" << num_voxel <<
        " exceeds max value of uimol_t:" << max_uimol_t << std::endl;
      std::cout << "Reduce lattice lengths or change uimol_t type in " <<
        "Common.hpp. Exiting now..." << std::endl;
      exit(0);
    }
  set_offsets();
  set_volume_structure();
  set_surface_structure();
}

thrust::device_vector<mol_t>& Compartment::get_offsets() {
  return offsets_;
}

void Compartment::set_offsets() {
  //col=even, layer=even
  offsets_[0] = -1;
  offsets_[1] = 1;
  offsets_[2] = -NUM_ROW-1;
  offsets_[3] = -NUM_ROW;
  offsets_[4] = NUM_ROW-1;
  offsets_[5] = NUM_ROW;
  offsets_[6] = -NUM_COLROW-NUM_ROW;
  offsets_[7] = -NUM_COLROW-1;
  offsets_[8] = -NUM_COLROW;
  offsets_[9] = NUM_COLROW-NUM_ROW;
  offsets_[10] = NUM_COLROW-1;
  offsets_[11] = NUM_COLROW;

  //col=even, layer=odd +24 = %layer*24
  offsets_[24] = -1;
  offsets_[25] = 1;
  offsets_[26] = -NUM_ROW;
  offsets_[27] = -NUM_ROW+1;
  offsets_[28] = NUM_ROW;
  offsets_[29] = NUM_ROW+1;
  offsets_[30] = -NUM_COLROW;
  offsets_[31] = -NUM_COLROW+1;
  offsets_[32] = -NUM_COLROW+NUM_ROW;
  offsets_[33] = NUM_COLROW;
  offsets_[34] = NUM_COLROW+1;
  offsets_[35] = NUM_COLROW+NUM_ROW;

  //col=odd, layer=even +12 = %col*12
  offsets_[12] = -1;
  offsets_[13] = 1;
  offsets_[14] = -NUM_ROW;
  offsets_[15] = -NUM_ROW+1;
  offsets_[16] = NUM_ROW;
  offsets_[17] = NUM_ROW+1;
  offsets_[18] = -NUM_COLROW-NUM_ROW;
  offsets_[19] = -NUM_COLROW;
  offsets_[20] = -NUM_COLROW+1;
  offsets_[21] = NUM_COLROW-NUM_ROW;
  offsets_[22] = NUM_COLROW;
  offsets_[23] = NUM_COLROW+1;

  //col=odd, layer=odd +36 = %col*12 + %layer*24
  offsets_[36] = -1;
  offsets_[37] = 1;
  offsets_[38] = -NUM_ROW-1;
  offsets_[39] = -NUM_ROW;
  offsets_[40] = NUM_ROW-1;
  offsets_[41] = NUM_ROW;
  offsets_[42] = -NUM_COLROW-1;
  offsets_[43] = -NUM_COLROW; //a
  offsets_[44] = -NUM_COLROW+NUM_ROW;
  offsets_[45] = NUM_COLROW-1;
  offsets_[46] = NUM_COLROW;
  offsets_[47] = NUM_COLROW+NUM_ROW;
}

const uint3& Compartment::get_lattice_dimensions() const {
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
  for(uimol_t i(0); i != get_lattice().get_num_voxel(); ++i) {
    get_volume_species().push_host_mol(i);
  }
  */
}


uimol_t Compartment::umol_to_uimol(const umol_t& umol) {
  return umol;
  /*
  return 
    umol.y + 
    umol.x*lattice_dimensions_.y + 
    umol.z*lattice_dimensions_.x*lattice_dimensions_.y;
    */
}

umol_t Compartment::uimol_to_umol(const uimol_t& uimol) {
  return uimol;
  /*
  const unsigned xy(lattice_dimensions_.x*lattice_dimensions_.y);
  return make_uint3(
      uimol%xy/lattice_dimensions_.y,
      uimol%xy%lattice_dimensions_.y,
      uimol/xy);
      */
}



void Compartment::set_surface_structure() {
  /*
  for (uimol_t i(0); i != NUM_COLROW; ++i) {
    get_surface_species().push_host_mol(i);
    get_surface_species().push_host_mol(NUM_VOXEL-1-i);
  }
  for (uimol_t i(1); i != NUM_LAY-1; ++i) {
    //layer_row yz-plane
    for (uimol_t j(0); j != NUM_ROW; ++j) {
      get_surface_species().push_host_mol(i*NUM_COLROW+j);
      get_surface_species().push_host_mol(i*NUM_COLROW+j+NUM_ROW*
                                         (NUM_COL-1));
    }
    //layer_col xz-plane
    for (uimol_t j(1); j != NUM_COL-1; ++j) {
      get_surface_species().push_host_mol(i*NUM_COLROW+j*NUM_ROW);
      get_surface_species().push_host_mol(i*NUM_COLROW+j*NUM_ROW+
                                         NUM_ROW-1);
    }
  }
  get_surface_species().populate_in_lattice();
  */
}


