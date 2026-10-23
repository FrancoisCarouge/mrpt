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

#include <mrpt/math/CMatrixF.h>
#include <mrpt/math/distributions.h>
#include <mrpt/math/matrix_serialization.h>
#include <mrpt/math/wrap2pi.h>
#include <mrpt/poses/CPoint2DPDFGaussian.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DPDF.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/random.h>
#include <mrpt/serialization/CArchive.h>
#include <mrpt/serialization/CSchemeArchiveBase.h>
#include <mrpt/system/os.h>

#include <Eigen/Dense>

// Strongly typed linear algebra: TypedLinearAlgebra over Eigen with mp-units elements
#include <mp-units/systems/si.h>

#include <fcarouge/eigen.hpp>
#include <fcarouge/mp_units.hpp>
#include <utility>

namespace
{
// A 2D pose, (x, y) in metres and phi in radians, and its covariance: the position block in square
// metres, the orientation variance in square radians and the cross terms in metre radians. The
// Jacobians of the pose operations map a pose change to a pose change. A product or sum whose units
// do not agree fails to compile.
using Length = mp_units::quantity<mp_units::si::metre, double>;
using Angle = mp_units::quantity<mp_units::si::radian, double>;
constexpr auto metre = mp_units::si::metre;
constexpr auto radian = mp_units::si::radian;
constexpr auto one = mp_units::one;

using Pose = fcarouge::typed_column_vector<Eigen::Matrix<double, 3, 1>, Length, Length, Angle>;

// the typed matrix of a typed expression, held in an Eigen matrix
template <typename E>
using Evaluated = fcarouge::typed_matrix<
    Eigen::Matrix<double, E::rows, E::columns>,
    typename E::row_indexes,
    typename E::column_indexes>;
using Covariance =
    Evaluated<decltype(std::declval<Pose>() * fcarouge::transposed(std::declval<Pose>()))>;
using Jacobian = Evaluated<decltype(std::declval<Pose>() / std::declval<Pose>())>;

Covariance typed(const mrpt::math::CMatrixDouble33& cov)
{
  return Covariance{Eigen::Matrix3d{cov.asEigen()}};
}

mrpt::math::CMatrixDouble33 untyped(const Covariance& cov)
{
  return mrpt::math::CMatrixDouble33(cov.data());
}

// the Jacobian of a rotation by phi of the position, the orientation unchanged
Jacobian rotation(const double phi, const double sign = 1.0)
{
  const double c = std::cos(phi), s = std::sin(phi);
  Jacobian J{};
  J.at<0, 0>(c * one);
  J.at<0, 1>(-sign * s * one);
  J.at<1, 0>(sign * s * one);
  J.at<1, 1>(c * one);
  J.at<2, 2>(1.0 * one);
  return J;
}

// the Jacobians of x1 (-) x0 with respect to x1 and to x0
std::pair<Jacobian, Jacobian> inverseCompositionJacobians(
    const mrpt::poses::CPose2D& x1, const mrpt::poses::CPose2D& x0)
{
  const double c = std::cos(x0.phi()), s = std::sin(x0.phi());
  const Length dx = (x1.x() - x0.x()) * metre;
  const Length dy = (x1.y() - x0.y()) * metre;

  // the rotation by -phi0
  const Jacobian dh_x1 = rotation(x0.phi(), -1.0);

  Jacobian dh_x0{};
  dh_x0.at<0, 0>(-c * one);
  dh_x0.at<0, 1>(-s * one);
  dh_x0.at<0, 2>((-dx * s + dy * c) / (1.0 * radian));
  dh_x0.at<1, 0>(s * one);
  dh_x0.at<1, 1>(-c * one);
  dh_x0.at<1, 2>((-dx * c - dy * s) / (1.0 * radian));
  dh_x0.at<2, 2>(-1.0 * one);
  return {dh_x1, dh_x0};
}
}  // namespace

using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::math;
using namespace mrpt::random;
using namespace mrpt::system;
using namespace std;

IMPLEMENTS_SERIALIZABLE(CPosePDFGaussian, CPosePDF, mrpt::poses)

/*---------------------------------------------------------------
  Constructor
  ---------------------------------------------------------------*/
CPosePDFGaussian::CPosePDFGaussian() : mean(0, 0, 0), cov() {}
/*---------------------------------------------------------------
  Constructor
  ---------------------------------------------------------------*/
CPosePDFGaussian::CPosePDFGaussian(const CPose2D& init_Mean, const CMatrixDouble33& init_Cov) :
    mean(init_Mean), cov(init_Cov)
{
}

/*---------------------------------------------------------------
  Constructor
  ---------------------------------------------------------------*/
CPosePDFGaussian::CPosePDFGaussian(const CPose2D& init_Mean) : mean(init_Mean), cov()
{
  cov.setZero();
}

uint8_t CPosePDFGaussian::serializeGetVersion() const { return 2; }
void CPosePDFGaussian::serializeTo(mrpt::serialization::CArchive& out) const
{
  out << mean;
  mrpt::math::serializeSymmetricMatrixTo(cov, out);
}
void CPosePDFGaussian::serializeFrom(mrpt::serialization::CArchive& in, uint8_t version)
{
  switch (version)
  {
    case 2:
    {
      in >> mean;
      mrpt::math::deserializeSymmetricMatrixFrom(cov, in);
    }
    break;
    case 1:
    {
      in >> mean;
      CMatrixFloat33 m;
      mrpt::math::deserializeSymmetricMatrixFrom(m, in);
      cov = m.cast_double();
    }
    break;
    case 0:
    {
      CMatrixF auxCov;
      in >> mean >> auxCov;
      cov = auxCov.cast_double();
    }
    break;
    default:
      MRPT_THROW_UNKNOWN_SERIALIZATION_VERSION(version);
  };
}

void CPosePDFGaussian::serializeTo(mrpt::serialization::CSchemeArchiveBase& out) const
{
  SCHEMA_SERIALIZE_DATATYPE_VERSION(1);
  out["mean"] = mean;
  out["cov"] = CMatrixD(cov);
}
void CPosePDFGaussian::serializeFrom(mrpt::serialization::CSchemeArchiveBase& in)
{
  int version = 0;
  SCHEMA_DESERIALIZE_DATATYPE_VERSION();
  switch (version)
  {
    case 1:
    {
      in["mean"].readTo(mean);
      CMatrixD m;
      in["cov"].readTo(m);
      cov = m;
    }
    break;
    default:
      MRPT_THROW_UNKNOWN_SERIALIZATION_VERSION(version);
  }
}

/*---------------------------------------------------------------
            copyFrom
  ---------------------------------------------------------------*/
void CPosePDFGaussian::copyFrom(const CPosePDF& o)
{
  if (this == &o) return;  // It may be used sometimes

  // Convert to gaussian pdf:
  o.getMean(mean);
  o.getCovariance(cov);
}

/*---------------------------------------------------------------
            copyFrom 3D
  ---------------------------------------------------------------*/
void CPosePDFGaussian::copyFrom(const CPose3DPDF& o)
{
  // Convert to gaussian pdf:
  mean = CPose2D(o.getMeanVal());
  CMatrixDouble66 C;
  o.getCovariance(C);

  // Clip to 3x3:
  C(2, 0) = C(0, 2) = C(0, 3);
  C(2, 1) = C(1, 2) = C(1, 3);
  C(2, 2) = C(3, 3);
  cov = C.block<3, 3>(0, 0);
}

/*---------------------------------------------------------------

  ---------------------------------------------------------------*/
bool CPosePDFGaussian::saveToTextFile(const std::string& file) const
{
  FILE* f = os::fopen(file.c_str(), "wt");
  if (!f)
  {
    return false;
  }

  os::fprintf(f, "%f %f %f\n", mean.x(), mean.y(), mean.phi());

  os::fprintf(f, "%f %f %f\n", cov(0, 0), cov(0, 1), cov(0, 2));
  os::fprintf(f, "%f %f %f\n", cov(1, 0), cov(1, 1), cov(1, 2));
  os::fprintf(f, "%f %f %f\n", cov(2, 0), cov(2, 1), cov(2, 2));

  os::fclose(f);
  return true;
}

/*---------------------------------------------------------------
            changeCoordinatesReference
 ---------------------------------------------------------------*/
void CPosePDFGaussian::changeCoordinatesReference(const CPose3D& newReferenceBase_)
{
  const CPose2D newReferenceBase = CPose2D(newReferenceBase_);

  // The mean:
  mean.composeFrom(newReferenceBase, mean);

  // The covariance:
  rotateCov(newReferenceBase.phi());
}

/*---------------------------------------------------------------
            changeCoordinatesReference
 ---------------------------------------------------------------*/
void CPosePDFGaussian::changeCoordinatesReference(const CPose2D& newReferenceBase)
{
  // The mean:
  mean.composeFrom(newReferenceBase, mean);
  // The covariance:
  rotateCov(newReferenceBase.phi());
}

/*---------------------------------------------------------------
            changeCoordinatesReference
 ---------------------------------------------------------------*/
void CPosePDFGaussian::rotateCov(const double ang)
{
  const Jacobian rot = rotation(ang);
  cov = untyped(Covariance{rot * typed(cov) * fcarouge::transposed(rot)});
}

/*---------------------------------------------------------------
          drawSingleSample
 ---------------------------------------------------------------*/
void CPosePDFGaussian::drawSingleSample(CPose2D& outPart) const
{
  MRPT_START

  CVectorDouble v;
  getRandomGenerator().drawGaussianMultivariate(v, cov);

  outPart.x(mean.x() + v[0]);
  outPart.y(mean.y() + v[1]);
  outPart.phi(mean.phi() + v[2]);

  // Range -pi,pi
  outPart.normalizePhi();

  MRPT_END_WITH_CLEAN_UP(cov.saveToTextFile("__DEBUG_EXC_DUMP_drawSingleSample_COV.txt"););
}

/*---------------------------------------------------------------
          drawManySamples
 ---------------------------------------------------------------*/
void CPosePDFGaussian::drawManySamples(size_t N, std::vector<CVectorDouble>& outSamples) const
{
  MRPT_START

  std::vector<CVectorDouble> rndSamples;

  getRandomGenerator().drawGaussianMultivariateMany(rndSamples, N, cov);
  outSamples.resize(N);
  for (size_t i = 0; i < N; i++)
  {
    outSamples[i].resize(3);
    outSamples[i][0] = mean.x() + rndSamples[i][0];
    outSamples[i][1] = mean.y() + rndSamples[i][1];
    outSamples[i][2] = mean.phi() + rndSamples[i][2];

    wrapToPiInPlace(outSamples[i][2]);
  }

  MRPT_END
}

/*---------------------------------------------------------------
          bayesianFusion
 ---------------------------------------------------------------*/
void CPosePDFGaussian::bayesianFusion(
    const CPosePDF& p1_,
    const CPosePDF& p2_,
    [[maybe_unused]] const double minMahalanobisDistToDrop)
{
  MRPT_START

  ASSERT_(p1_.GetRuntimeClass() == CLASS_ID(CPosePDFGaussian));
  ASSERT_(p2_.GetRuntimeClass() == CLASS_ID(CPosePDFGaussian));

  const auto* p1 = dynamic_cast<const CPosePDFGaussian*>(&p1_);
  const auto* p2 = dynamic_cast<const CPosePDFGaussian*>(&p2_);

  const CMatrixDouble33 C1 = p1->cov;
  const CMatrixDouble33 C2 = p2->cov;

  const CMatrixDouble33 C1_inv = C1.inverse_LLt();
  const CMatrixDouble33 C2_inv = C2.inverse_LLt();

  auto x1 = CMatrixDouble31(p1->mean);
  auto x2 = CMatrixDouble31(p2->mean);

  CMatrixDouble33 auxC = C1_inv + C2_inv;
  this->cov = auxC.inverse_LLt();
  this->enforceCovSymmetry();

  auto x = CMatrixDouble31(this->cov.asEigen() * (C1_inv * x1 + C2_inv * x2));

  this->mean.x(x(0, 0));
  this->mean.y(x(1, 0));
  this->mean.phi(x(2, 0));
  this->mean.normalizePhi();

  MRPT_END
}

/*---------------------------------------------------------------
          inverse
 ---------------------------------------------------------------*/
void CPosePDFGaussian::inverse(CPosePDF& o) const
{
  ASSERT_(o.GetRuntimeClass() == CLASS_ID(CPosePDFGaussian));
  auto* out = dynamic_cast<CPosePDFGaussian*>(&o);

  // The mean:
  out->mean = CPose2D(0, 0, 0) - mean;

  // The covariance:
  const double ccos = ::cos(mean.phi());
  const double ssin = ::sin(mean.phi());
  const Length x = mean.x() * metre;
  const Length y = mean.y() * metre;

  // jacobian:
  Jacobian H{};
  H.at<0, 0>(-ccos * one);
  H.at<0, 1>(-ssin * one);
  H.at<0, 2>((x * ssin - y * ccos) / (1.0 * radian));
  H.at<1, 0>(ssin * one);
  H.at<1, 1>(-ccos * one);
  H.at<1, 2>((x * ccos + y * ssin) / (1.0 * radian));
  H.at<2, 2>(-1.0 * one);

  out->cov = untyped(Covariance{H * typed(cov) * fcarouge::transposed(H)});
}

/*---------------------------------------------------------------
              +=
 ---------------------------------------------------------------*/
void CPosePDFGaussian::operator+=(const CPose2D& Ap)
{
  // The covariance of mean (+) Ap, Ap being certain: J * COV * J^T, J the
  // Jacobian of the composition with respect to its first pose.
  CMatrixDouble33 df_dx(UNINITIALIZED_MATRIX), df_du(UNINITIALIZED_MATRIX);
  CPosePDF::jacobiansPoseComposition(mean, Ap, df_dx, df_du, true, false);
  const CMatrixDouble33 OLD_COV = cov;
  cov = mrpt::math::multiply_HCHt(df_dx, OLD_COV);

  mean = mean + Ap;
}

/*---------------------------------------------------------------
            evaluatePDF
 ---------------------------------------------------------------*/
double CPosePDFGaussian::evaluatePDF(const CPose2D& x) const
{
  auto X = CMatrixDouble31(x);
  auto MU = CMatrixDouble31(mean);

  return math::normalPDF(X, MU, this->cov);
}

/*---------------------------------------------------------------
            evaluateNormalizedPDF
 ---------------------------------------------------------------*/
double CPosePDFGaussian::evaluateNormalizedPDF(const CPose2D& x) const
{
  auto X = CMatrixDouble31(x);
  auto MU = CMatrixDouble31(mean);

  return math::normalPDF(X, MU, this->cov) / math::normalPDF(MU, MU, this->cov);
}

/*---------------------------------------------------------------
            enforceCovSymmetry
 ---------------------------------------------------------------*/
void CPosePDFGaussian::enforceCovSymmetry()
{
  // Differences, when they exist, appear in the ~15'th significant
  //  digit, so... just take one of them arbitrarily!
  cov(0, 1) = cov(1, 0);
  cov(0, 2) = cov(2, 0);
  cov(1, 2) = cov(2, 1);
}

/*---------------------------------------------------------------
            mahalanobisDistanceTo
 ---------------------------------------------------------------*/
double CPosePDFGaussian::mahalanobisDistanceTo(const CPosePDFGaussian& theOther)
{
  MRPT_START

  auto MU = CVectorFixedDouble<3>(mean);
  MU -= CVectorFixedDouble<3>(theOther.mean);

  wrapToPiInPlace(MU[2]);

  if (MU[0] == 0 && MU[1] == 0 && MU[2] == 0)
    return 0;  // This is the ONLY case where we know the result, whatever
  // COVs are.

  CMatrixDouble33 COV_ = cov;
  COV_ += theOther.cov;

  const CMatrixDouble33 COV_inv = COV_.inverse_LLt();

  // (~MU) * (!COV_) * MU
  return std::sqrt(mrpt::math::multiply_HtCH_scalar(MU.asEigen(), COV_inv));

  MRPT_END
}

void CPosePDFGaussian::printTo(std::ostream& out) const { out << *this; }

/*---------------------------------------------------------------
            operator <<
 ---------------------------------------------------------------*/
std::ostream& mrpt::poses::operator<<(std::ostream& out, const CPosePDFGaussian& obj)
{
  out << "Mean: " << obj.mean << "\n";
  out << "Covariance:\n" << obj.cov << "\n";

  return out;
}

/*---------------------------------------------------------------
            operator +
 ---------------------------------------------------------------*/
poses::CPosePDFGaussian mrpt::poses::operator+(
    const mrpt::poses::CPose2D& A, const mrpt::poses::CPosePDFGaussian& B)
{
  poses::CPosePDFGaussian ret(B);
  ret.changeCoordinatesReference(A);
  return ret;
}

/*---------------------------------------------------------------
            assureMinCovariance
 ---------------------------------------------------------------*/
void CPosePDFGaussian::assureMinCovariance(double minStdXY, double minStdPhi)
{
  cov(0, 0) = max(cov(0, 0), square(minStdXY));
  cov(1, 1) = max(cov(1, 1), square(minStdXY));
  cov(2, 2) = max(cov(2, 2), square(minStdPhi));
}

/*---------------------------------------------------------------
            inverseComposition
  Set 'this' = 'x' - 'ref', computing the mean using the "-"
  operator and the covariances through the corresponding Jacobians.
 ---------------------------------------------------------------*/
void CPosePDFGaussian::inverseComposition(const CPosePDFGaussian& xv, const CPosePDFGaussian& xi)
{
  // COV:
  //  Y = dh_xv * XV * dh_xv^T  + dh_xi * XI * dh_xi^T
  const auto [dh_xv, dh_xi] = inverseCompositionJacobians(xv.mean, xi.mean);
  this->cov = untyped(
      Covariance{
          dh_xv * typed(xv.cov) * fcarouge::transposed(dh_xv) +
          dh_xi * typed(xi.cov) * fcarouge::transposed(dh_xi)});

  // Mean:
  mean = xv.mean - xi.mean;
}

/*---------------------------------------------------------------
            inverseComposition
  Set \f$ this = x1 \ominus x0 \f$ , computing the mean using
   the "-" operator and the covariances through the corresponding
  Jacobians (Given the 3x3 cross-covariance matrix of variables x0 and x0).
 ---------------------------------------------------------------*/
void CPosePDFGaussian::inverseComposition(
    const CPosePDFGaussian& x1, const CPosePDFGaussian& x0, const CMatrixDouble33& COV_01)
{
  // Build the cov:
  //  Y = dh_xv * XV * dh_xv^T  + dh_xi * XI * dh_xi^T +  A + At
  //     being A = dh_dx0 * COV01 * dh_dx1^t
  const auto [dh_x1, dh_x0] = inverseCompositionJacobians(x1.mean, x0.mean);
  const Covariance A{dh_x0 * typed(COV_01) * fcarouge::transposed(dh_x1)};
  this->cov = untyped(
      Covariance{
          dh_x0 * typed(x0.cov) * fcarouge::transposed(dh_x0) +
          dh_x1 * typed(x1.cov) * fcarouge::transposed(dh_x1) + A + fcarouge::transposed(A)});

  // mean
  mean = x1.mean - x0.mean;
}

/*---------------------------------------------------------------
              +=
 ---------------------------------------------------------------*/
void CPosePDFGaussian::operator+=(const CPosePDFGaussian& Ap)
{
  // COV:
  const CMatrixDouble33 OLD_COV = this->cov;
  CMatrixDouble33 df_dx(UNINITIALIZED_MATRIX), df_du(UNINITIALIZED_MATRIX);

  CPosePDF::jacobiansPoseComposition(
      this->mean,  // x
      Ap.mean,     // u
      df_dx, df_du);

  this->cov = mrpt::math::multiply_HCHt(df_dx, OLD_COV) + mrpt::math::multiply_HCHt(df_du, Ap.cov);

  // MEAN:
  this->mean = this->mean + Ap.mean;
}

/** Returns the PDF of the 2D point \f$ g = q \oplus l\f$ with "q"=this pose and
 * "l" a point without uncertainty */
void CPosePDFGaussian::composePoint(const mrpt::math::TPoint2D& l, CPoint2DPDFGaussian& g) const
{
  // Mean:
  double gx, gy;
  mean.composePoint(l.x, l.y, gx, gy);
  g.mean.x(gx);
  g.mean.y(gy);

  // COV:
  CMatrixDouble33 df_dx(UNINITIALIZED_MATRIX), df_du(UNINITIALIZED_MATRIX);
  CPosePDF::jacobiansPoseComposition(
      this->mean,              // x
      CPose2D(l.x, l.y, 0.0),  // u: the point, as a pose
      df_dx, df_du,
      true,  // Eval df_dx
      false  // Eval df_du (not needed)
  );

  const auto dp_dx = CMatrixFixed<double, 2, 3>(df_dx.block<2, 3>(0, 0));
  g.cov = dp_dx * this->cov * dp_dx.transpose();
}

bool mrpt::poses::operator==(const CPosePDFGaussian& p1, const CPosePDFGaussian& p2)
{
  return p1.mean == p2.mean && p1.cov == p2.cov;
}

CPosePDFGaussian mrpt::poses::operator+(const CPosePDFGaussian& a, const CPosePDFGaussian& b)
{
  CPosePDFGaussian res(a);
  res += b;
  return res;
}

CPosePDFGaussian mrpt::poses::operator-(const CPosePDFGaussian& a, const CPosePDFGaussian& b)
{
  CPosePDFGaussian res;
  res.inverseComposition(a, b);
  return res;
}
