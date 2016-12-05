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


#ifndef __VisualLogger_hpp
#define __VisualLogger_hpp

#include <fstream>
#include <Common.hpp>

class VisualLogger
{
public:
  VisualLogger(Model&);
  ~VisualLogger() {}
  void fire();
  void initialize();
  void push_species(Species&);
private:
  void initialize_log();
  void log_structure_species();
  void log_species();
  void log_mols(const umol_t);
private:
  umol_t marker_;
  std::string filename_;
  std::ofstream logfile_;
  Compartment& compartment_;
  Stepper& stepper_;
  std::vector<Species*> species_;
};

#endif /* __VisualLogger_hpp */
