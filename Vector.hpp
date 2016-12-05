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


#ifndef __Vector_hpp
#define __Vector_hpp

template<typename T>
struct Vector
{
  Vector(const T a=0, const T b=0, const T c=0);
  Vector<T> operator + (const T val) const;
  Vector<T> operator - (const T val) const;
  Vector<T> operator * (const T val) const;
  Vector<T> operator / (const T val) const;
  const Vector<T>& operator += (const T val);
  const Vector<T>& operator -= (const T val);
  const Vector<T>& operator *= (const T val);
  const Vector<T>& operator /= (const T val);
  Vector<T> operator + (const Vector<T>& vector) const;
  Vector<T> operator - (const Vector<T>& vector) const;
  Vector<T> operator * (const Vector<T>& vector) const;
  Vector<T> operator / (const Vector<T>& vector) const;
  const Vector<T>& operator += (const Vector<T>& vector);
  const Vector<T>& operator -= (const Vector<T>& vector);
  const Vector<T>& operator *= (const Vector<T>& vector);
  const Vector<T>& operator /= (const Vector<T>& vector);
  T x;
  T y;
  T z;
};

template<class T>
Vector<T>::Vector(const T a, const T b, const T c)
  : x(a),
    y(b),
    z(c) {}

template<class T>
Vector<T> Vector<T>::operator + (const T val) const {
  return Vector<T>(x+val, y+val, z+val);
}

template<class T>
Vector<T> Vector<T>::operator - (const T val) const {
  return Vector<T>(x-val, y-val, z-val);
}

template<class T>
Vector<T> Vector<T>::operator * (const T val) const {
  return Vector<T>(x*val, y*val, z*val);
}

template<class T>
Vector<T> Vector<T>::operator / (const T val) const {
  return Vector<T>(x/val, y/val, z/val);
}

template<class T>
const Vector<T>& Vector<T>::operator += (const T val) {
  x += val;
  y += val;
  z += val;
  return *this;
}

template<class T>
const Vector<T>& Vector<T>::operator -= (const T val) {
  x -= val;
  y -= val;
  z -= val;
  return *this;
}

template<class T>
const Vector<T>& Vector<T>::operator *= (const T val) {
  x *= val;
  y *= val;
  z *= val;
  return *this;
}

template<class T>
const Vector<T>& Vector<T>::operator /= (const T val) {
  x /= val;
  y /= val;
  z /= val;
  return *this;
}

template<class T>
Vector<T> Vector<T>::operator + (const Vector<T>& vector) const {
  return Vector<T>(x+vector.x, y+vector.y, z+vector.z);
}

template<class T>
Vector<T> Vector<T>::operator - (const Vector<T>& vector) const {
  return Vector<T>(x-vector.x, y-vector.y, z-vector.z);
}

template<class T>
Vector<T> Vector<T>::operator * (const Vector<T>& vector) const {
  return Vector<T>(x*vector.x, y*vector.y, z*vector.z);
}

template<class T>
Vector<T> Vector<T>::operator / (const Vector<T>& vector) const {
  return Vector<T>(x/vector.x, y/vector.y, z/vector.z);
}

template<class T>
const Vector<T>& Vector<T>::operator += (const Vector<T>& vector) {
  x += vector.x;
  y += vector.y;
  z += vector.z;
  return *this;
}

template<class T>
const Vector<T>& Vector<T>::operator -= (const Vector<T>& vector) {
  x -= vector.x;
  y -= vector.y;
  z -= vector.z;
  return *this;
}

template<class T>
const Vector<T>& Vector<T>::operator *= (const Vector<T>& vector) {
  x *= vector.x;
  y *= vector.y;
  z *= vector.z;
  return *this;
}

template<class T>
const Vector<T>& Vector<T>::operator /= (const Vector<T>& vector) {
  x /= vector.x;
  y /= vector.y;
  z /= vector.z;
  return *this;
}

#endif /* __Vector_hpp */
