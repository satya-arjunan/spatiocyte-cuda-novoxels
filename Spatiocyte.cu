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

#include <iostream> 
#include <boost/date_time/posix_time/posix_time.hpp>
#include <Spatiocyte.hpp>
#include <Model.hpp>
#include <Reaction.hpp>
#include <VisualLogger.hpp>

int main() {
  Model model;
  Species A("A", 53687091, 1e-12, model, model.get_compartment(),
            model.get_compartment().get_volume_species());
  /*
  Species B("B", 800256, 1e-12, model, model.get_compartment(),
            model.get_compartment().get_volume_species());
  Species C("C", 0, 1e-12, model, model.get_compartment(),
            model.get_compartment().get_volume_species());
  Reaction AB_to_C;
  AB_to_C.push_substrate(A);
  AB_to_C.push_substrate(B);
  AB_to_C.push_product(C);
  AB_to_C.set_p(1);
  */
  model.initialize();
  /*
  VisualLogger visual_logger(model);
  */
  model.get_stepper().push_diffuser(A.get_diffuser());
  /*
  model.get_stepper().push_diffuser(B.get_diffuser());
  model.get_stepper().push_diffuser(C.get_diffuser());
  model.get_stepper().set_visual_logger(visual_logger);
  visual_logger.push_species(A);
  visual_logger.push_species(B);
  visual_logger.push_species(C);
  //visual_logger.push_species(model.get_compartment().get_surface_species());
  //visual_logger.push_species(model.get_compartment().get_volume_species());
  visual_logger.initialize();
  */


  model.run(0.0001);
  boost::posix_time::ptime start(
      boost::posix_time::microsec_clock::universal_time()); 
  //model.run(0.1);
  unsigned steps(10000);
  //unsigned steps(model.run(0.5));
  model.step(steps);
  cudaDeviceSynchronize();
  boost::posix_time::ptime end(
      boost::posix_time::microsec_clock::universal_time());
  boost::posix_time::time_duration duration(end-start);
  double bups((A.get_mols().size())*steps/
               (duration.total_milliseconds()/1000.0));
  std::cout <<  bups/1e+9 << " BUPS, " << 
    bups*sizeof(umol_t)*2/(1024*1024*1024) <<
    " GB/s, " << duration.total_milliseconds() << " ms" << std::endl;
}
