/*                    _
                     | |    Mobile Robot Programming Toolkit (MRPT)
 _ __ ___  _ __ _ __ | |_
| '_ ` _ \| '__| '_ \| __|          https://www.mrpt.org/
| | | | | | |  | |_) | |_
|_| |_| |_|_|  | .__/ \__|     https://github.com/MRPT/mrpt/
               | |
               |_|

 Copyright (c) 2005-2026, Individual contributors, see AUTHORS file
 See: https://www.mrpt.org/Authors - All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
*/
#pragma once

// Typed linear algebra of the pose covariances: every element of a covariance and of a Jacobian
// has its physical unit, so that a propagation J Σ Jᵀ is checked for units at compile time, and every
// Jacobian is a sparse matrix of its structural zeros, so that only the products they admit are
// computed. Private to mrpt_poses: the public interfaces keep their untyped matrices.

#include <mrpt/math/CMatrixFixed.h>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "fcarouge/sparse_linear_algebra.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "mp-units/framework/quantity.h"
#include "mp-units/systems/si.h"

namespace fcarouge
{
/** MRPT's adapter of the typed linear algebra to mp-units elements: a quantity element is stored as
 * its numerical value in its own unit, and a stored value is read back as a quantity of that unit. */
template <typename To, mp_units::Quantity From>
struct element_caster<To, From>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> To
  {
    static_assert(std::same_as<typename From::rep, std::remove_cvref_t<To>>);
    return value.numerical_value_in(From::unit);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From>
{
  [[nodiscard]] static constexpr auto operator()(From value) -> To
  {
    static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
    return value * To::reference;
  }
};
}  // namespace fcarouge

namespace mrpt::poses::typed
{
template <auto Reference>
using Quantity = mp_units::quantity<Reference, double>;
using Length = Quantity<mp_units::si::metre>;
using Angle = Quantity<mp_units::si::radian>;
using InverseLength = Quantity<mp_units::one / mp_units::si::metre>;
using InverseAngle = Quantity<mp_units::one / mp_units::si::radian>;
using Ratio = Quantity<mp_units::one>;

/** The elements of a 2D pose, (x, y, φ), and of a 2D point, (x, y). */
using Pose2 = std::tuple<Length, Length, Angle>;
using InversePose2 = std::tuple<InverseLength, InverseLength, InverseAngle>;
using Point2 = std::tuple<Length, Length>;
using InversePoint2 = std::tuple<InverseLength, InverseLength>;
using Point3 = std::tuple<Length, Length, Length>;
using InversePoint3 = std::tuple<InverseLength, InverseLength, InverseLength>;

/** The elements of a 3D pose, (x, y, z, yaw, pitch, roll), and of a 3D pose with its orientation as a
 * quaternion, (x, y, z, qr, qx, qy, qz). */
using Pose3 = std::tuple<Length, Length, Length, Angle, Angle, Angle>;
using InversePose3 =
    std::tuple<InverseLength, InverseLength, InverseLength, InverseAngle, InverseAngle, InverseAngle>;
using PoseQuat = std::tuple<Length, Length, Length, Ratio, Ratio, Ratio, Ratio>;
using InversePoseQuat =
    std::tuple<InverseLength, InverseLength, InverseLength, Ratio, Ratio, Ratio, Ratio>;

template <std::size_t Rows, std::size_t Columns>
inline constexpr fcarouge::sparse::pattern dense{fcarouge::sparse::dense<Rows, Columns>};

/** The covariance of a pose, m², m·rad, rad², and of a point, m². */
using Pose2Covariance =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, dense<3, 3>>, Pose2, Pose2>;
using Point2Covariance =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, dense<2, 2>>, Point2, Point2>;
using Point3Covariance =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, dense<3, 3>>, Point3, Point3>;
/** The information of a pose, the inverse of its covariance: 1/m², 1/(m·rad), 1/rad². */
using Pose2Information =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, dense<3, 3>>, InversePose2, InversePose2>;
/** The covariance of a 3D pose, m², m·rad, rad², and with a quaternion, m², m, [1]. */
using Pose3Covariance =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, dense<6, 6>>, Pose3, Pose3>;
using PoseQuatCovariance =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, dense<7, 7>>, PoseQuat, PoseQuat>;

/** The patterns of the Jacobians, the derivatives of a pose or a point by a pose. */
/** A rotation of the position, the orientation unchanged. */
inline constexpr fcarouge::sparse::pattern rotation{
    fcarouge::sparse::shape<3, 3>, {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}}};
/** A translation by a rotated lever arm: the identity and the derivative of the position by the
 * orientation. */
inline constexpr fcarouge::sparse::pattern lever_arm{
    fcarouge::sparse::shape<3, 3>, {{0, 0}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};
/** Both: a rotated position and its derivative by the orientation. */
inline constexpr fcarouge::sparse::pattern rotation_lever_arm{
    fcarouge::sparse::shape<3, 3>, {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 2}}};
/** The derivative of a composed point by the pose. */
inline constexpr fcarouge::sparse::pattern point_lever_arm{
    fcarouge::sparse::shape<2, 3>, {{0, 0}, {0, 2}, {1, 1}, {1, 2}}};

/** The derivative of a composed 3D pose by its first pose: the identity of the translation, the
 * derivative of the translation by the orientation, and of the orientation by the orientation. */
inline constexpr fcarouge::sparse::pattern pose3_by_first{
    fcarouge::sparse::shape<6, 6>, [](std::size_t row, std::size_t column)
    { return (row < 3 and column == row) or column >= 3; }};
/** The derivative of a composed 3D pose by its second pose: the rotation of the translation, and the
 * derivative of the orientation by the orientation. */
inline constexpr fcarouge::sparse::pattern pose3_by_second{
    fcarouge::sparse::shape<6, 6>,
    [](std::size_t row, std::size_t column) { return (row < 3) == (column < 3); }};
/** The derivative of a 3D pose by its quaternion form: the identity of the translation, and the
 * derivative of the angles by the quaternion. */
inline constexpr fcarouge::sparse::pattern euler_by_quaternion{
    fcarouge::sparse::shape<6, 7>, [](std::size_t row, std::size_t column)
    { return (row < 3 and column == row) or (row >= 3 and column >= 3); }};

/** The derivative of a composed quaternion pose by its first pose: the identity of the translation, and
 * the derivatives by the quaternion. */
inline constexpr fcarouge::sparse::pattern quat_by_first{
    fcarouge::sparse::shape<7, 7>, [](std::size_t row, std::size_t column)
    { return (row < 3 and column == row) or column >= 3; }};
/** The derivative of a composed quaternion pose by its second pose: the rotation of the translation,
 * and the derivative of the quaternion by the quaternion. */
inline constexpr fcarouge::sparse::pattern quat_by_second{
    fcarouge::sparse::shape<7, 7>,
    [](std::size_t row, std::size_t column) { return (row < 3) == (column < 3); }};
/** The derivative of the inverse of a quaternion pose: the translation by the whole pose, and the
 * quaternion by the quaternion. */
inline constexpr fcarouge::sparse::pattern quat_inverse{
    fcarouge::sparse::shape<7, 7>,
    [](std::size_t row, std::size_t column) { return row < 3 or column >= 3; }};
/** The derivative of a quaternion pose by its Euler form: the identity of the translation, and the
 * derivative of the quaternion by the angles. */
inline constexpr fcarouge::sparse::pattern quaternion_by_euler{
    fcarouge::sparse::shape<7, 6>, [](std::size_t row, std::size_t column)
    { return (row < 3 and column == row) or (row >= 3 and column >= 3); }};

template <auto Pattern, typename RowIndexes, typename ColumnIndexes>
using Jacobian =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, Pattern>, RowIndexes, ColumnIndexes>;
template <auto Pattern>
using Pose3Jacobian = Jacobian<Pattern, Pose3, InversePose3>;
template <auto Pattern>
using PoseQuatJacobian = Jacobian<Pattern, PoseQuat, InversePoseQuat>;

/** The transposed inverse of a pose Jacobian, which carries an information matrix to the new frame:
 * Λ' = J⁻ᵀ Λ J⁻¹. */
template <auto Pattern>
using Pose2InformationJacobian = Jacobian<Pattern, InversePose2, Pose2>;
/** A rotation of a point. */
using Point3Rotation = Jacobian<dense<3, 3>, Point3, InversePoint3>;
using Point2Rotation = Jacobian<dense<2, 2>, Point2, InversePoint2>;

template <auto Pattern>
using Pose2Jacobian =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, Pattern>, Pose2, InversePose2>;
template <auto Pattern>
using Point2Jacobian =
    fcarouge::typed_matrix<fcarouge::sparse_matrix<double, Pattern>, Point2, InversePose2>;

/** The typed matrix from the numerical values of its stored elements, in row-major order, each in
 * the unit of its element. */
template <typename Typed, std::size_t Count>
[[nodiscard]] inline Typed make(const std::array<double, Count>& values)
{
  static constexpr auto structure{Typed::matrix::structure};
  static_assert(Count == structure.count);
  Typed result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>)
  {
    (result.template at<structure.positions[Index].row, structure.positions[Index].column>(
         values[Index] *
         fcarouge::tla::element_at<
             Typed, structure.positions[Index].row,
             structure.positions[Index].column>::reference),
     ...);
  }(std::make_index_sequence<Count>{});
  return result;
}

/** The typed matrix of the stored elements of an untyped one, each read in the unit of its
 * element. The structural zeros are not read: a debug build asserts that they are zeros. */
template <typename Typed, mrpt::math::matrix_dim_t Rows, mrpt::math::matrix_dim_t Columns>
[[nodiscard]] inline Typed typed(const mrpt::math::CMatrixFixed<double, Rows, Columns>& value)
{
  static constexpr auto structure{Typed::matrix::structure};
  std::array<double, structure.count> values;
  [&]<std::size_t... Index>(std::index_sequence<Index...>)
  {
    ((values[Index] = value(structure.positions[Index].row, structure.positions[Index].column)),
     ...);
  }(std::make_index_sequence<structure.count>{});
#ifndef NDEBUG
  for (std::size_t row = 0; row < Rows; ++row)
  {
    for (std::size_t column = 0; column < Columns; ++column)
    {
      assert(structure.stores(row, column) or value(row, column) == 0.);
    }
  }
#endif
  return make<Typed>(values);
}

/** The untyped matrix of a typed one, each element in its own unit. */
template <typename Typed>
[[nodiscard]] inline auto untyped(const Typed& value)
{
  mrpt::math::CMatrixFixed<double, Typed::rows, Typed::columns> result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>)
  {
    ((result(Index / Typed::columns, Index % Typed::columns) =
          value.template at<Index / Typed::columns, Index % Typed::columns>().numerical_value_in(
              fcarouge::tla::element_at<Typed, Index / Typed::columns,
                                        Index % Typed::columns>::unit)),
     ...);
  }(std::make_index_sequence<Typed::rows * Typed::columns>{});
  return result;
}
}  // namespace mrpt::poses::typed
