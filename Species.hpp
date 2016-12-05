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


#ifndef __Species_hpp
#define __Species_hpp

#include <thrust/device_vector.h>
#include <Common.hpp>
#include <Diffuser.hpp>

class Species
{ 
public: 
  Species(const std::string, const unsigned, const double, Model&, Compartment&,
      Species& vacant, const bool is_structure_species=false);
  ~Species() {}
  void initialize();
  void populate();
  void populate_in_lattice();
  void push_host_mol(const umol_t);
  void push_reaction(Reaction&);
  bool is_structure_species() const;
  bool is_root_structure_species() const;
  voxel_t get_id() const;
  voxel_t get_vac_id() const;
  Diffuser& get_diffuser();
  Species& get_vacant();
  Model& get_model() const;
  Compartment& get_compartment();
  const std::string& get_name() const;
  const std::string get_name_id() const;
  std::vector<umol_t>& get_host_mols();
  thrust::device_vector<umol_t>& get_mols();
  std::vector<Reaction*>& get_reactions();
private:
  const std::string get_init_name(const std::string) const;
private:
  Compartment& compartment_;
  Model& model_;
  Species& vacant_;
  const std::string name_;
  const unsigned init_nmols_;
  const bool is_structure_species_;
  const voxel_t id_;
  const voxel_t vac_id_;
  std::vector<umol_t> host_mols_;
  thrust::device_vector<umol_t> mols_;
  Diffuser diffuser_;
  std::vector<Reaction*> reactions_;
};

#endif /* __Species_hpp */

