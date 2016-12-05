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

#include <Stepper.hpp>
#include <Diffuser.hpp>
#include <VisualLogger.hpp>

void Stepper::step() {
  for(unsigned i(0); i != diffusers_.size(); ++i) {
    diffusers_[i]->walk();
  }
  //visual_logger_->fire();
  /*
  time_ += 1;
  if(!(unsigned(time_)%10))
    {
      visual_logger_->fire();
    }
    */
}

void Stepper::set_diffuser(Diffuser& diffuser) {
  diffuser_ = &diffuser;
}

void Stepper::push_diffuser(Diffuser& diffuser) {
  diffusers_.push_back(&diffuser);
}

void Stepper::set_visual_logger(VisualLogger& visual_logger) {
  visual_logger_ = &visual_logger;
}

double Stepper::get_current_time() const {
  return time_;
}

