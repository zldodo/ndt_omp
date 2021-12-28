#include "ndt_omp.hpp"
/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_REGISTRATION_NDT_OMP_IMPL_H_
#define PCL_REGISTRATION_NDT_OMP_IMPL_H_

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

template<typename PointT , typename T>
pcl::PointCloud<PointT> transformPointCloud(
    const pcl::PointCloud<PointT> & cloud_in,
    const T & transform)
{
  pcl::PointCloud<PointT> cloud_out;
  transformPointCloud(cloud_in, cloud_out, transform);
  return cloud_out;
}

template<typename PointSource>
Eigen::Vector3d point_to_vector3d(const PointSource & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget>
pclomp::NormalDistributionsTransform<PointSource, PointTarget>::NormalDistributionsTransform()
: target_cells_(),
  resolution_(1.0f),
  step_size_(0.1),
  outlier_ratio_(0.55),
  gauss_d1_(),
  gauss_d2_(),
  gauss_d3_(),
  trans_probability_()
{
  reg_name_ = "NormalDistributionsTransform";

  double gauss_c1, gauss_c2;

  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  gauss_c1 = 10.0 * (1 - outlier_ratio_);
  gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  gauss_d3_ = -log(gauss_c2);
  gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3_;
  gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3_) / gauss_d1_);

  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;

  search_method = DIRECT7;
  num_threads_ = omp_get_max_threads();
}

std::tuple<double, double, double> initDistributionParams(
  const double outlier_ratio, const double resolution)
{
  // Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
  const double c1 = 10 * (1 - outlier_ratio);
  const double c2 = outlier_ratio / pow(resolution, 3);
  const double d3 = -log(c2);
  const double d1 = -log(c1 + c2) - d3;
  const double d2 = -2 * log((-log(c1 * exp(-0.5) + c2) - d3) / d1);
  return {d1, d2, d3};
}

Eigen::Matrix4d makeTransformation(const Vector6d & v)
{
  const Eigen::Translation<double, 3> t(v(0), v(1), v(2));
  const Eigen::AngleAxis<double> rx(v(3), Eigen::Vector3d::UnitX());
  const Eigen::AngleAxis<double> ry(v(4), Eigen::Vector3d::UnitY());
  const Eigen::AngleAxis<double> rz(v(5), Eigen::Vector3d::UnitZ());
  return (t * rx * ry * rz).matrix();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget>
void
pclomp::NormalDistributionsTransform<PointSource, PointTarget>::computeTransformation(
  PointCloudSource & output, const Eigen::Matrix4f & guess)
{
  nr_iterations_ = 0;
  converged_ = false;

  std::tie(gauss_d1_, gauss_d2_, gauss_d3_) = initDistributionParams(outlier_ratio_, resolution_);

  // Initialise final transformation to the guessed one
  final_transformation_ = guess;
  // Apply guessed transformation prior to search for neighbours

  Eigen::Transform<double, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
  eig_transformation.matrix() = final_transformation_.template cast<double>();

  // Convert initial guess matrix to 6 element transformation vector
  Vector6d p;
  p << eig_transformation.translation(), eig_transformation.rotation().eulerAngles(0, 1, 2);

  Matrix6d hessian;

  double score = 0;
  double delta_p_norm;

  // Calculate derivatives of initial transform vector, subsequent derivative calculations are done in the step length determination.
  Vector6d score_gradient;
  std::tie(score_gradient, hessian, score) = computeDerivatives(output, p);

  output = transformPointCloud(output, guess);
  transformation_array_.clear();
  transformation_array_.push_back(final_transformation_);
  while (!converged_) {
    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson 2009]
    const Eigen::JacobiSVD<Matrix6d> sv(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    Vector6d delta_p = sv.solve(-score_gradient);

    //Calculate step length with guaranteed sufficient decrease [More, Thuente 1994]
    delta_p_norm = delta_p.norm();

    if (delta_p_norm == 0 || delta_p_norm != delta_p_norm) {
      trans_probability_ = score / static_cast<double>(input_->size());
      converged_ = delta_p_norm == delta_p_norm;
      return;
    }

    delta_p.normalize();
    delta_p_norm = computeStepLengthMT(
      p, delta_p, delta_p_norm, step_size_,
      transformation_epsilon_ / 2, score, score_gradient, hessian,
      output);
    delta_p *= delta_p_norm;

    transformation_ = makeTransformation(delta_p).cast<float>();

    transformation_array_.push_back(final_transformation_);

    p = p + delta_p;

    // Update Visualizer (untested)
    if (update_visualizer_ != 0) {
      update_visualizer_(output, std::vector<int>(), *target_, std::vector<int>() );
    }

    if (nr_iterations_ > max_iterations_ ||
      (nr_iterations_ && (std::fabs(delta_p_norm) < transformation_epsilon_)))
    {
      converged_ = true;
    }

    nr_iterations_++;

  }

  // Store transformation probability. The relative differences within each scan registration are accurate
  // but the normalization constants need to be modified for it to be globally accurate
  trans_probability_ = score / static_cast<double>(input_->size());

  hessian_ = hessian;
}

#ifndef _OPENMP
int omp_get_max_threads() {return 1;}
int omp_get_thread_num() {return 0;}
#endif

Eigen::Matrix<double, 4, 6> computePointGradient(
  const Eigen::Vector3d & x,
  const Eigen::Matrix<double, 8, 3> & j_ang)
{
  // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector p.
  // Derivative w.r.t. ith element of transform vector corresponds to column i,
  // Equation 6.18 and 6.19 [Magnusson 2009]
  Eigen::Matrix<double, 8, 1> y = j_ang * x;

  Eigen::Matrix<double, 4, 6> gradient = Eigen::Matrix<double, 4, 6>::Zero();
  gradient.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();

  gradient(1, 3) = y(0);
  gradient(2, 3) = y(1);
  gradient(0, 4) = y(2);
  gradient(1, 4) = y(3);
  gradient(2, 4) = y(4);
  gradient(0, 5) = y(5);
  gradient(1, 5) = y(6);
  gradient(2, 5) = y(7);

  return gradient;
}

Eigen::Matrix<double, 24, 6> computePointHessian(
  const Eigen::Vector3d & x,
  const Eigen::Matrix<double, 15, 3> & h_ang)
{
  Eigen::Matrix<double, 15, 1> y = h_ang * x;

  // Vectors from Equation 6.21 [Magnusson 2009]
  Eigen::Vector4d a(0, y(0), y(1), 0.0f);
  Eigen::Vector4d b(0, y(2), y(3), 0.0f);
  Eigen::Vector4d c(0, y(4), y(5), 0.0f);
  Eigen::Vector4d d(y(6), y(7), y(8), 0.0f);
  Eigen::Vector4d e(y(9), y(10), y(11), 0.0f);
  Eigen::Vector4d f(y(12), y(13), y(14), 0.0f);

  Eigen::Matrix<double, 24, 6> hessian = Eigen::Matrix<double, 24, 6>::Zero();
  // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform vector p.
  // Derivative w.r.t. ith and jth elements of transform vector corresponds to
  // the 3x1 block matrix starting at (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
  hessian.block<4, 1>(12, 3) = a;
  hessian.block<4, 1>(16, 3) = b;
  hessian.block<4, 1>(20, 3) = c;
  hessian.block<4, 1>(12, 4) = b;
  hessian.block<4, 1>(16, 4) = d;
  hessian.block<4, 1>(20, 4) = e;
  hessian.block<4, 1>(12, 5) = c;
  hessian.block<4, 1>(16, 5) = e;
  hessian.block<4, 1>(20, 5) = f;
  return hessian;
}

double computeScoreIncrement(
  const Eigen::Vector3d & x_trans, const Eigen::Matrix3d & c_inv,
  const double d1, const double d2)
{
  Eigen::Matrix<double, 1, 4> x_trans4(x_trans[0], x_trans[1], x_trans[2], 0.0f);
  Eigen::Matrix4d c_inv4 = Eigen::Matrix4d::Zero();
  c_inv4.topLeftCorner(3, 3) = c_inv;
  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  const double e_x_cov_x = exp(-d2 * x_trans4.dot(x_trans4 * c_inv4) * 0.5f);
  // Calculate probability of transformed points existence, Equation 6.9 [Magnusson 2009]
  return -d1 * e_x_cov_x;
}

Vector6d computeScoreGradient(
  const Eigen::Matrix<double, 4, 6> & point_gradient4,
  const Eigen::Vector3d & x_trans, const Eigen::Matrix3d & c_inv,
  const double d1, const double d2)
{
  Eigen::Matrix<double, 1, 4> x_trans4(x_trans[0], x_trans[1], x_trans[2], 0.0f);
  Eigen::Matrix4d c_inv4 = Eigen::Matrix4d::Zero();
  c_inv4.topLeftCorner(3, 3) = c_inv;

  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  const double e_x_cov_x = exp(-d2 * x_trans4.dot(x_trans4 * c_inv4) * 0.5f);
  // Calculate probability of transformed points existence, Equation 6.9 [Magnusson 2009]
  const double score_inc = -d1 * e_x_cov_x;

  // Update gradient, Equation 6.12 [Magnusson 2009]
  const Eigen::Matrix<double, 6, 1> g = x_trans4 * c_inv4 * point_gradient4;

  return d1 * d2 * e_x_cov_x * g;
}

Matrix6d computeScoreHessian(
  const Eigen::Matrix<double, 4, 6> & point_gradient4,
  const Eigen::Matrix<double, 24, 6> & point_hessian_,
  const Eigen::Vector3d & x_trans, const Eigen::Matrix3d & c_inv,
  const double d1, const double d2)
{
  Eigen::Matrix<double, 1, 4> x_trans4(x_trans[0], x_trans[1], x_trans[2], 0.0f);
  Eigen::Matrix4d c_inv4 = Eigen::Matrix4d::Zero();
  c_inv4.topLeftCorner(3, 3) = c_inv;

  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  const double e_x_cov_x = exp(-d2 * x_trans4.dot(x_trans4 * c_inv4) * 0.5f);

  const Eigen::Matrix<double, 6, 1> g = x_trans4 * c_inv4 * point_gradient4;
  const Eigen::Matrix<double, 1, 4> xc = x_trans4 * c_inv4;
  const Eigen::Matrix<double, 6, 6> m = point_gradient4.transpose() * c_inv4 * point_gradient4;

  Matrix6d hessian = Matrix6d::Zero();
  for (int i = 0; i < 6; i++) {
    // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
    const Eigen::Matrix<double, 6, 1> h = xc * point_hessian_.block<4, 6>(i * 4, 0);

    for (int j = 0; j < hessian.cols(); j++) {
      // Update hessian, Equation 6.13 [Magnusson 2009]
      hessian(i, j) += d1 * d2 * e_x_cov_x * (-d2 * g(i) * g(j) + h(j) + m(j, i));
    }
  }

  return hessian;
}

double round_cos(const double angle)
{
  if (fabs(angle) < 10e-5) {
    return 1.0;
  }
  return cos(angle);
}

double round_sin(const double angle)
{
  if (fabs(angle) < 10e-5) {
    return 0.0;
  }
  return sin(angle);
}

Eigen::Matrix<double, 8, 3> computeAngularGradient(const Eigen::Vector3d & angles)
{
  const double cx = round_cos(angles(0));
  const double sx = round_sin(angles(0));
  const double cy = round_cos(angles(1));
  const double sy = round_sin(angles(1));
  const double cz = round_cos(angles(2));
  const double sz = round_sin(angles(2));

  Eigen::Matrix<double, 8, 3> j_ang;
  j_ang.row(0) << -sx * sz + cx * sy * cz, -sx * cz - cx * sy * sz, -cx * cy;
  j_ang.row(1) << cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy;
  j_ang.row(2) << -sy * cz, sy * sz, cy;
  j_ang.row(3) << sx * cy * cz, -sx * cy * sz, sx * sy;
  j_ang.row(4) << -cx * cy * cz, cx * cy * sz, -cx * sy;
  j_ang.row(5) << -cy * sz, -cy * cz, 0;
  j_ang.row(6) << cx * cz - sx * sy * sz, -cx * sz - sx * sy * cz, 0;
  j_ang.row(7) << sx * cz + cx * sy * sz, cx * sy * cz - sx * sz, 0;
  return j_ang;
}

Eigen::Matrix<double, 15, 3> computeAngularHessian(const Eigen::Vector3d & angles)
{
  const double cx = round_cos(angles(0));
  const double sx = round_sin(angles(0));
  const double cy = round_cos(angles(1));
  const double sy = round_sin(angles(1));
  const double cz = round_cos(angles(2));
  const double sz = round_sin(angles(2));

  Eigen::Matrix<double, 15, 3> h_ang;
  h_ang.row(0) << -cx * sz - sx * sy * cz, -cx * cz + sx * sy * sz, sx * cy;    // a2
  h_ang.row(1) << -sx * sz + cx * sy * cz, -cx * sy * sz - sx * cz, -cx * cy;   // a3

  h_ang.row(2) << cx * cy * cz, -cx * cy * sz, cx * sy;                   // b2
  h_ang.row(3) << sx * cy * cz, -sx * cy * sz, sx * sy;                   // b3

  h_ang.row(4) << -sx * cz - cx * sy * sz, sx * sz - cx * sy * cz, 0;     // c2
  h_ang.row(5) << cx * cz - sx * sy * sz, -sx * sy * cz - cx * sz, 0;     // c3

  h_ang.row(6) << -cy * cz, cy * sz, sy;                                  // d1
  h_ang.row(7) << -sx * sy * cz, sx * sy * sz, sx * cy;                   // d2
  h_ang.row(8) << cx * sy * cz, -cx * sy * sz, -cx * cy;                  // d3

  h_ang.row(9) << sy * sz, sy * cz, 0;                                    // e1
  h_ang.row(10) << -sx * cy * sz, -sx * cy * cz, 0;                       // e2
  h_ang.row(11) << cx * cy * sz, cx * cy * cz, 0;                         // e3

  h_ang.row(12) << -cy * cz, cy * sz, 0;                                  // f1
  h_ang.row(13) << -cx * sz - sx * sy * cz, -cx * cz + sx * sy * sz, 0;   // f2
  h_ang.row(14) << -sx * sz + cx * sy * cz, -cx * sy * sz - sx * cz, 0;   // f3

  return h_ang;
}

template<typename PointSource, typename PointTarget>
std::tuple<Vector6d, Matrix6d, double>
pclomp::NormalDistributionsTransform<PointSource, PointTarget>::computeDerivatives(
  const PointCloudSource & input_cloud,
  const Vector6d & p,
  const bool compute_hessian)
{
  const PointCloudSource trans_cloud = transformPointCloud(input_cloud, makeTransformation(p));

  std::vector<double> scores(input_->size());
  std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> gradients(input_->size());
  std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> hessians(input_->size());

  const Eigen::Matrix<double, 8, 3> j_ang = computeAngularGradient(p.tail(3));
  const Eigen::Matrix<double, 15, 3> h_ang = computeAngularHessian(p.tail(3));

  // Update gradient and hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (std::size_t idx = 0; idx < input_->size(); idx++) {
    int thread_n = omp_get_thread_num();

    const PointSource x_trans_pt = trans_cloud.at(idx);

    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;

    // Find neighbors (Radius search has been experimentally faster than direct neighbor checking.
    switch (search_method) {
      case KDTREE:
        target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);
        break;
      case DIRECT26:
        target_cells_.getNeighborhoodAtPoint(x_trans_pt, neighborhood);
        break;
      default:
      case DIRECT7:
        target_cells_.getNeighborhoodAtPoint7(x_trans_pt, neighborhood);
        break;
      case DIRECT1:
        target_cells_.getNeighborhoodAtPoint1(x_trans_pt, neighborhood);
        break;
    }

    const Eigen::Vector3d x_trans_ref = point_to_vector3d(x_trans_pt);

    double s = 0;
    for (const TargetGridLeafConstPtr cell : neighborhood) {
      const Eigen::Vector3d x_trans = x_trans_ref - cell->getMean();
      const Eigen::Matrix3d c_inv = cell->getInverseCov();
      s += computeScoreIncrement(x_trans, c_inv, gauss_d1_, gauss_d2_);
    }
    scores[idx] = s;

    const Eigen::Vector3d x = point_to_vector3d(input_->at(idx));
    const Eigen::Matrix<double, 4, 6> pg = computePointGradient(x, j_ang);
    const Eigen::Matrix<double, 24, 6> ph = computePointHessian(x, h_ang);

    Vector6d g = Vector6d::Zero();
    for (const TargetGridLeafConstPtr cell : neighborhood) {
      const Eigen::Vector3d x_trans = x_trans_ref - cell->getMean();
      const Eigen::Matrix3d c_inv = cell->getInverseCov();
      g += computeScoreGradient(pg, x_trans, c_inv, gauss_d1_, gauss_d2_);
    }
    gradients[idx].noalias() = g;

    Matrix6d h = Matrix6d::Zero();
    for (const TargetGridLeafConstPtr cell : neighborhood) {
      const Eigen::Vector3d x_trans = x_trans_ref - cell->getMean();
      const Eigen::Matrix3d c_inv = cell->getInverseCov();
      h += computeScoreHessian(pg, ph, x_trans, c_inv, gauss_d1_, gauss_d2_);
    }
    hessians[idx].noalias() = h;
  }

  Vector6d gradient = Vector6d::Zero();
  Matrix6d hessian = Matrix6d::Zero();
  double score = 0;

  // Ensure that the result is invariant against the summing up order
  for (std::size_t i = 0; i < input_->size(); i++) {
    score += scores[i];
    gradient += gradients[i];
    hessian += hessians[i];
  }

  return {gradient, hessian, score};
}

Eigen::Matrix<double, 3, 6> computePointGradient(
  const Eigen::Matrix<double, 8, 3> & j_ang,
  const Eigen::Vector3d & x)
{
  Eigen::Matrix<double, 3, 6> point_gradient_ = Eigen::Matrix<double, 3, 6>::Zero();
  // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector p.
  // Derivative w.r.t. ith element of transform vector corresponds to column i,
  // Equation 6.18 and 6.19 [Magnusson 2009]
  point_gradient_(1, 3) = x.dot(j_ang.row(0));
  point_gradient_(2, 3) = x.dot(j_ang.row(1));
  point_gradient_(0, 4) = x.dot(j_ang.row(2));
  point_gradient_(1, 4) = x.dot(j_ang.row(3));
  point_gradient_(2, 4) = x.dot(j_ang.row(4));
  point_gradient_(0, 5) = x.dot(j_ang.row(5));
  point_gradient_(1, 5) = x.dot(j_ang.row(6));
  point_gradient_(2, 5) = x.dot(j_ang.row(7));
  return point_gradient_;
}

Eigen::Matrix<double, 18, 6> computePointHessian(
  const Eigen::Matrix<double, 15, 3> & h_ang,
  const Eigen::Vector3d & x)
{
  // Vectors from Equation 6.21 [Magnusson 2009]
  const Eigen::Vector3d a(0, x.dot(h_ang.row(0)), x.dot(h_ang.row(1)));
  const Eigen::Vector3d b(0, x.dot(h_ang.row(2)), x.dot(h_ang.row(3)));
  const Eigen::Vector3d c(0, x.dot(h_ang.row(4)), x.dot(h_ang.row(5)));
  const Eigen::Vector3d d(x.dot(h_ang.row(6)), x.dot(h_ang.row(7)), x.dot(h_ang.row(8)));
  const Eigen::Vector3d e(x.dot(h_ang.row(9)), x.dot(h_ang.row(10)), x.dot(h_ang.row(11)));
  const Eigen::Vector3d f(x.dot(h_ang.row(12)), x.dot(h_ang.row(13)), x.dot(h_ang.row(14)));

  // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform vector p.
  // Derivative w.r.t. ith and jth elements of transform vector corresponds to
  // the 3x1 block matrix starting at (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
  Eigen::Matrix<double, 18, 6> point_hessian_ = Eigen::Matrix<double, 18, 6>::Zero();
  point_hessian_.block<3, 1>(9, 3) = a;
  point_hessian_.block<3, 1>(12, 3) = b;
  point_hessian_.block<3, 1>(15, 3) = c;
  point_hessian_.block<3, 1>(9, 4) = b;
  point_hessian_.block<3, 1>(12, 4) = d;
  point_hessian_.block<3, 1>(15, 4) = e;
  point_hessian_.block<3, 1>(9, 5) = c;
  point_hessian_.block<3, 1>(12, 5) = e;
  point_hessian_.block<3, 1>(15, 5) = f;
  return point_hessian_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget>
Matrix6d pclomp::NormalDistributionsTransform<PointSource, PointTarget>::computeHessian(
  const Eigen::Matrix<double, 8, 3> & j_ang,
  const Eigen::Matrix<double, 15, 3> & h_ang,
  const PointCloudSource & trans_cloud) const
{
  // Initialize Point Gradient and Hessian

  Matrix6d hessian = Matrix6d::Zero();

  // Precompute Angular Derivatives unnecessary because only used after regular derivative calculation

  // Update hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]
  for (size_t idx = 0; idx < input_->size(); idx++) {
    const PointSource x_trans_pt = trans_cloud.at(idx);

    // Find neighbors (Radius search has been experimentally faster than direct neighbor checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    switch (search_method) {
      case KDTREE:
        target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);
        break;
      case DIRECT26:
        target_cells_.getNeighborhoodAtPoint(x_trans_pt, neighborhood);
        break;
      default:
      case DIRECT7:
        target_cells_.getNeighborhoodAtPoint7(x_trans_pt, neighborhood);
        break;
      case DIRECT1:
        target_cells_.getNeighborhoodAtPoint1(x_trans_pt, neighborhood);
        break;
    }

    const Eigen::Vector3d x = point_to_vector3d(input_->at(idx));
    const Eigen::Vector3d x_trans_ref = point_to_vector3d(x_trans_pt);
    for (const TargetGridLeafConstPtr cell : neighborhood) {
      // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
      const Eigen::Vector3d x_trans = x_trans_ref - cell->getMean();
      // Uses precomputed covariance for speed.
      const Eigen::Matrix3d c_inv = cell->getInverseCov();

      Eigen::Matrix<double, 3, 6> point_gradient_;
      Eigen::Matrix<double, 18, 6> point_hessian_;
      point_gradient_.setZero();
      point_gradient_.block<3, 3>(0, 0).setIdentity();
      point_hessian_.setZero();

      // Compute derivative of transform function w.r.t. transform vector,
      // J_E and H_E in Equations 6.18 and 6.20 [Magnusson 2009]
      point_gradient_ = computePointGradient(j_ang, x);
      point_hessian_ = computePointHessian(h_ang, x);
      // Update hessian, lines 21 in Algorithm 2,
      // according to Equations 6.10, 6.12 and 6.13, respectively [Magnusson 2009]
      updateHessian(hessian, point_gradient_, point_hessian_, x_trans, c_inv);
    }
  }
  return hessian;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget>
void
pclomp::NormalDistributionsTransform<PointSource, PointTarget>::updateHessian(
  Matrix6d & hessian,
  const Eigen::Matrix<double, 3, 6> & point_gradient_,
  const Eigen::Matrix<double, 18, 6> & point_hessian_,
  const Eigen::Vector3d & x_trans,
  const Eigen::Matrix3d & c_inv) const
{
  Eigen::Vector3d cov_dxd_pi;
  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  double e_x_cov_x = gauss_d2_ * exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x) {
    return;
  }

  // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
  e_x_cov_x *= gauss_d1_;

  for (int i = 0; i < 6; i++) {
    // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
    cov_dxd_pi = c_inv * point_gradient_.col(i);

    for (int j = 0; j < hessian.cols(); j++) {
      // Update hessian, Equation 6.13 [Magnusson 2009]
      hessian(i, j) += e_x_cov_x *
        (-gauss_d2_ * x_trans.dot(cov_dxd_pi) * x_trans.dot(c_inv * point_gradient_.col(j)) +
        x_trans.dot(c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
        point_gradient_.col(j).dot(cov_dxd_pi) );
    }
  }
}

std::tuple<bool, std::pair<double, double>, std::pair<double, double>, std::pair<double, double>>
updateIntervalMT(
  const std::pair<double, double> & a,
  const std::pair<double, double> & f,
  const std::pair<double, double> & g,
  double a_t, double f_t, double g_t)
{
  const auto [a_l, a_u] = a;
  const auto [f_l, f_u] = f;
  const auto [g_l, g_u] = g;
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
  if (f_t > f_l) {
    return {false, std::make_pair(a_l, a_t), std::make_pair(f_l, f_t), std::make_pair(g_l, g_t)};
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
  if (g_t * (a_l - a_t) > 0) {
    return {false, std::make_pair(a_t, a_u), std::make_pair(f_t, f_u), std::make_pair(g_t, g_u)};
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
  if (g_t * (a_l - a_t) < 0) {
    return {false, std::make_pair(a_t, a_l), std::make_pair(f_t, f_l), std::make_pair(g_t, g_l)};
  }
  // Interval Converged
  return {true, a, f, g};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double trialValueSelectionMT(
  const std::pair<double, double> & a,
  const std::pair<double, double> & f,
  const std::pair<double, double> & g,
  const double a_t, const double f_t, const double g_t)
{
  const auto [a_l, a_u] = a;
  const auto [f_l, f_u] = f;
  const auto [g_l, g_u] = g;
  // Case 1 in Trial Value Selection [More, Thuente 1994]
  if (f_t > f_l) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
    // Equation 2.4.2 [Sun, Yuan 2006]
    double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs(a_c - a_l) < std::fabs(a_q - a_l)) {
      return a_c;
    } else {
      return 0.5 * (a_q + a_c);
    }
  }
  // Case 2 in Trial Value Selection [More, Thuente 1994]
  if (g_t * g_l < 0) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs(a_c - a_t) >= std::fabs(a_s - a_t)) {
      return a_c;
    } else {
      return a_s;
    }
  }
  // Case 3 in Trial Value Selection [More, Thuente 1994]
  if (std::fabs(g_t) <= std::fabs(g_l)) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs(a_c - a_t) < std::fabs(a_s - a_t)) {
      a_t_next = a_c;
    } else {
      a_t_next = a_s;
    }

    if (a_t > a_l) {
      return std::min(a_t + 0.66 * (a_u - a_t), a_t_next);
    } else {
      return std::max(a_t + 0.66 * (a_u - a_t), a_t_next);
    }
  }
  // Case 4 in Trial Value Selection [More, Thuente 1994]
  // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
  // Equation 2.4.52 [Sun, Yuan 2006]
  double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
  double w = std::sqrt(z * z - g_t * g_u);
  // Equation 2.4.56 [Sun, Yuan 2006]
  return a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget>
double
pclomp::NormalDistributionsTransform<PointSource, PointTarget>::computeStepLengthMT(
  const Vector6d & x, Vector6d & step_dir, double step_init,
  double step_max,
  double step_min, double & score, Vector6d & score_gradient,
  Matrix6d & hessian,
  PointCloudSource & trans_cloud)
{
  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  Vector6d x_t;

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0) {
      return 0;
    } else {
      // Reverse step direction and calculate optimal step.
      d_phi_0 *= -1;
      step_dir *= -1;

    }
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decrease constant, Equation 1.1 [More, Thuete 1994]
  double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval, Equation 2.1 [More, Thuente 1994]
  double f_l = auxiliaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxiliaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxiliaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxiliaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making step_min == step_max
  bool interval_converged = (step_max - step_min) < 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  x_t = x + step_dir * a_t;

  // Updates score, gradient and hessian.  Hessian calculation is unnecessary but testing showed that most step calculations use the
  // initial step suggestion and recalculation the reusable portions of the hessian would intail more computation time.
  std::tie(score_gradient, hessian, score) = computeDerivatives(*input_, x_t, true);

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxiliaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxiliaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergence or a value satisfies the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More, Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations &&
    !(psi_t <= 0 /*Sufficient Decrease*/ && d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/))
  {
    // Use auxiliary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(
        std::make_pair(a_l, a_u),
        std::make_pair(f_l, f_u),
        std::make_pair(g_l, g_u),
        a_t, psi_t, d_psi_t);
    } else {
      a_t = trialValueSelectionMT(
        std::make_pair(a_l, a_u),
        std::make_pair(f_l, f_u),
        std::make_pair(g_l, g_u),
        a_t, phi_t, d_phi_t);
    }

    a_t = std::clamp(a_t, step_min, step_max);

    x_t = x + step_dir * a_t;

    // Updates score, gradient. Values stored to prevent wasted computation.
    std::tie(score_gradient, hessian, score) = computeDerivatives(*input_, x_t, false);

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxiliaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxiliaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l = f_l + phi_0 - mu * d_phi_0 * a_l;
      g_l = g_l + mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u = f_u + phi_0 - mu * d_phi_0 * a_u;
      g_u = g_u + mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      const auto [converged, a, f, g] = updateIntervalMT(
        std::make_pair(a_l, a_u),
        std::make_pair(f_l, f_u),
        std::make_pair(g_l, g_u),
        a_t, psi_t, d_psi_t);
      interval_converged = converged;
      std::tie(a_l, a_u) = a;
      std::tie(f_l, f_u) = f;
      std::tie(g_l, g_u) = g;
    } else {
      // Update interval end points using Modified Updating Algorithm [More, Thuente 1994]
      const auto [converged, a, f, g] = updateIntervalMT(
        std::make_pair(a_l, a_u),
        std::make_pair(f_l, f_u),
        std::make_pair(g_l, g_u),
        a_t, phi_t, d_phi_t);
      interval_converged = converged;
      std::tie(a_l, a_u) = a;
      std::tie(f_l, f_u) = f;
      std::tie(g_l, g_u) = g;
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnecessary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
  if (step_iterations) {
    // Precompute Angular Derivatives (eq. 6.19 and 6.21)[Magnusson 2009]
    const Eigen::Matrix<double, 8, 3> j_ang = computeAngularGradient(x_t.tail(3));
    const Eigen::Matrix<double, 15, 3> h_ang = computeAngularHessian(x_t.tail(3));
    hessian = computeHessian(j_ang, h_ang, trans_cloud);
  }

  trans_cloud = transformPointCloud(*input_, makeTransformation(x_t));
  final_transformation_ = makeTransformation(x_t).cast<float>();
  return a_t;
}


template<typename PointSource, typename PointTarget>
double pclomp::NormalDistributionsTransform<PointSource, PointTarget>::calculateScore(
  const PointCloudSource & trans_cloud) const
{
  double score = 0;

  for (const PointSource x_trans_pt : trans_cloud) {
    // Find neighbors (Radius search has been experimentally faster than direct neighbor checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    switch (search_method) {
      case KDTREE:
        target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);
        break;
      case DIRECT26:
        target_cells_.getNeighborhoodAtPoint(x_trans_pt, neighborhood);
        break;
      default:
      case DIRECT7:
        target_cells_.getNeighborhoodAtPoint7(x_trans_pt, neighborhood);
        break;
      case DIRECT1:
        target_cells_.getNeighborhoodAtPoint1(x_trans_pt, neighborhood);
        break;
    }

    const Eigen::Vector3d x_trans_ref = point_to_vector3d(x_trans_pt);
    for (const TargetGridLeafConstPtr cell : neighborhood) {
      // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
      const Eigen::Vector3d x_trans = x_trans_ref - cell->getMean();
      // Uses precomputed covariance for speed.
      const Eigen::Matrix3d c_inv = cell->getInverseCov();

      // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
      double e_x_cov_x = exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);
      // Calculate probability of transformed points existence, Equation 6.9 [Magnusson 2009]
      double score_inc = -gauss_d1_ * e_x_cov_x - gauss_d3_;

      score += score_inc / neighborhood.size();
    }
  }
  return (score) / static_cast<double>(trans_cloud.size());
}

#endif // PCL_REGISTRATION_NDT_IMPL_H_
