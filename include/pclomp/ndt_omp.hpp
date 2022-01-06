/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
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

#ifndef PCL_REGISTRATION_NDT_OMP_H_
#define PCL_REGISTRATION_NDT_OMP_H_

#include <pcl/registration/registration.h>
#include <pcl/search/impl/search.hpp>
#include "voxel_grid_covariance_omp.hpp"

#include <unsupported/Eigen/NonLinearOptimization>

namespace pclomp
{
enum NeighborSearchMethod
{
  KDTREE,
  DIRECT26,
  DIRECT7,
  DIRECT1
};

/** \brief A 3D Normal Distribution Transform registration implementation for point cloud data.
 * \note For more information please see
 * <b>Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform —
 * an Efficient Representation for Registration, Surface Analysis, and Loop Detection.
 * PhD thesis, Orebro University. Orebro Studies in Technology 36.</b>,
 * <b>More, J., and Thuente, D. (1994). Line Search Algorithm with Guaranteed Sufficient Decrease
 * In ACM Transactions on Mathematical Software.</b> and
 * Sun, W. and Yuan, Y, (2006) Optimization Theory and Methods: Nonlinear Programming. 89-100
 * \note Math refactored by Todor Stoyanov.
 * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
 */
template<typename PointSource, typename PointTarget>
class NormalDistributionsTransform : public pcl::Registration<PointSource, PointTarget>
{
protected:
  typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
  typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
  typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

  typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;
  typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
  typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

  typedef pcl::PointIndices::Ptr PointIndicesPtr;
  typedef pcl::PointIndices::ConstPtr PointIndicesConstPtr;

  /** \brief Typename of searchable voxel grid containing mean and covariance. */
  typedef pclomp::VoxelGridCovariance<PointTarget> TargetGrid;
  /** \brief Typename of pointer to searchable voxel grid. */
  typedef TargetGrid * TargetGridPtr;
  /** \brief Typename of const pointer to searchable voxel grid. */
  typedef const TargetGrid * TargetGridConstPtr;
  /** \brief Typename of const pointer to searchable voxel grid leaf. */
  typedef typename TargetGrid::LeafConstPtr TargetGridLeafConstPtr;

public:
#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  typedef pcl::shared_ptr<NormalDistributionsTransform<PointSource, PointTarget>> Ptr;
  typedef pcl::shared_ptr<const NormalDistributionsTransform<PointSource, PointTarget>> ConstPtr;
#else
  typedef boost::shared_ptr<NormalDistributionsTransform<PointSource, PointTarget>> Ptr;
  typedef boost::shared_ptr<const NormalDistributionsTransform<PointSource, PointTarget>> ConstPtr;
#endif


  /** \brief Constructor.
   * Sets \ref outlier_ratio_ to 0.35, \ref step_size_ to 0.05 and \ref resolution_ to 1.0
   */
  NormalDistributionsTransform();

  /** \brief Empty destructor */
  virtual ~NormalDistributionsTransform() {}

  void setNumThreads(int n)
  {
    num_threads_ = n;
  }

  inline int getNumThreads() const
  {
    return num_threads_;
  }

  /** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the input source to).
   * \param[in] cloud the input point cloud target
   */
  inline void
  setInputTarget(const PointCloudTargetConstPtr & cloud)
  {
    pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);
    init();
  }

  /** \brief Set/change the voxel grid resolution.
   * \param[in] resolution side length of voxels
   */
  inline void
  setResolution(double resolution)
  {
    // Prevents unnecessary voxel initiations
    if (resolution_ != resolution) {
      resolution_ = resolution;
      if (input_) {
        init();
      }
    }
  }

  /** \brief Get voxel grid resolution.
   * \return side length of voxels
   */
  inline double
  getResolution() const
  {
    return resolution_;
  }

  /** \brief Get the newton line search maximum step length.
   * \return maximum step length
   */
  inline double
  getStepSize() const
  {
    return step_size_;
  }

  /** \brief Set/change the newton line search maximum step length.
   * \param[in] step_size maximum step length
   */
  inline void
  setStepSize(double step_size)
  {
    step_size_ = step_size;
  }

  /** \brief Get the point cloud outlier ratio.
   * \return outlier ratio
   */
  inline double
  getOutlierRatio() const
  {
    return outlier_ratio_;
  }

  inline void setNeighborhoodSearchMethod(NeighborSearchMethod method)
  {
    search_method = method;
  }

  inline NeighborSearchMethod
  getNeighborhoodSearchMethod() const
  {
    return search_method;
  }

  /** \brief Get the registration alignment probability.
   * \return transformation probability
   */
  inline double
  getTransformationProbability() const
  {
    return trans_probability_;
  }

  /** \brief Get the number of iterations required to calculate alignment.
   * \return final number of iterations
   */
  inline int
  getFinalNumIteration() const
  {
    return nr_iterations_;
  }

  /** \brief Return the hessian matrix */
  inline Eigen::Matrix<double, 6, 6>
  getHessian() const
  {
    return hessian_;
  }

  /** \brief Return the transformation array */
  inline const std::vector<Eigen::Matrix4f>
  getFinalTransformationArray() const
  {
    return transformation_array_;
  }

  // negative log likelihood function
  // lower is better
  double calculateScore(const PointCloudSource & cloud) const;

protected:
  using pcl::Registration<PointSource, PointTarget>::reg_name_;
  using pcl::Registration<PointSource, PointTarget>::getClassName;
  using pcl::Registration<PointSource, PointTarget>::input_;
  using pcl::Registration<PointSource, PointTarget>::indices_;
  using pcl::Registration<PointSource, PointTarget>::target_;
  using pcl::Registration<PointSource, PointTarget>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget>::transformation_;
  using pcl::Registration<PointSource, PointTarget>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget>::converged_;
  using pcl::Registration<PointSource, PointTarget>::corr_dist_threshold_;
  using pcl::Registration<PointSource, PointTarget>::inlier_threshold_;

  using pcl::Registration<PointSource, PointTarget>::update_visualizer_;

  /** \brief Estimate the transformation and returns the transformed source (input) as output.
   * \param[out] output the resultant input transformed point cloud dataset
   */
  virtual void
  computeTransformation(PointCloudSource & output)
  {
    computeTransformation(output, Eigen::Matrix4f::Identity());
  }

  /** \brief Estimate the transformation and returns the transformed source (input) as output.
   * \param[out] output the resultant input transformed point cloud dataset
   * \param[in] guess the initial gross estimation of the transformation
   */
  virtual void
  computeTransformation(PointCloudSource & output, const Eigen::Matrix4f & guess);

  /** \brief Initiate covariance voxel structure. */
  void inline
  init()
  {
    target_cells_.setLeafSize(resolution_, resolution_, resolution_);
    target_cells_.setInputCloud(target_);
    // Initiate voxel structure.
    target_cells_.filter(true);
  }

  /** \brief Compute derivatives of probability function w.r.t. the transformation vector.
   * \note Equation 6.10, 6.12 and 6.13 [Magnusson 2009].
   * \param[out] score_gradient the gradient vector of the probability function w.r.t. the transformation vector
   * \param[out] hessian the hessian matrix of the probability function w.r.t. the transformation vector
   * \param[in] trans_cloud transformed point cloud
   * \param[in] p the current transform vector
   * \param[in] compute_hessian flag to calculate hessian, unnecessary for step calculation.
   */
  std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>, double>
  computeDerivatives(
    const PointCloudSource & trans_cloud,
    const Eigen::Matrix<double, 6, 1> & p,
    const bool compute_hessian = true) const;

  /** \brief Precompute angular components of derivatives.
   * \note Equation 6.19 and 6.21 [Magnusson 2009].
   * \param[in] p the current transform vector
   * \param[in] compute_hessian flag to calculate hessian, unnecessary for step calculation.
   */
  void
  computeAngleDerivatives(const Eigen::Matrix<double, 6, 1> & p);

  /** \brief Compute hessian of probability function w.r.t. the transformation vector.
   * \note Equation 6.13 [Magnusson 2009].
   * \param[out] hessian the hessian matrix of the probability function w.r.t. the transformation vector
   * \param[in] trans_cloud transformed point cloud
   * \param[in] p the current transform vector
   */

  Eigen::Matrix<double, 6, 6> computeHessian(
      const Eigen::Matrix<double, 8, 3> & j_ang,
      const Eigen::Matrix<double, 15, 3> & h_ang,
      const PointCloudSource & trans_cloud) const;

  /** \brief Compute line search step length and update transform and probability derivatives using More-Thuente method.
   * \note Search Algorithm [More, Thuente 1994]
   * \param[in] x initial transformation vector, \f$ x \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ \vec{p} \f$ in Algorithm 2 [Magnusson 2009]
   * \param[in] step_dir descent direction, \f$ p \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ \delta \vec{p} \f$ normalized in Algorithm 2 [Magnusson 2009]
   * \param[in] step_init initial step length estimate, \f$ \alpha_0 \f$ in Moore-Thuente (1994) and the normal of \f$ \delta \vec{p} \f$ in Algorithm 2 [Magnusson 2009]
   * \param[in] step_max maximum step length, \f$ \alpha_max \f$ in Moore-Thuente (1994)
   * \param[in] step_min minimum step length, \f$ \alpha_min \f$ in Moore-Thuente (1994)
   * \param[out] score final score function value, \f$ f(x + \alpha p) \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ score \f$ in Algorithm 2 [Magnusson 2009]
   * \param[in,out] score_gradient gradient of score function w.r.t. transformation vector, \f$ f'(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ \vec{g} \f$ in Algorithm 2 [Magnusson 2009]
   * \param[out] hessian hessian of score function w.r.t. transformation vector, \f$ f''(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ H \f$ in Algorithm 2 [Magnusson 2009]
   * \param[in,out] trans_cloud transformed point cloud, \f$ X \f$ transformed by \f$ T(\vec{p},\vec{x}) \f$ in Algorithm 2 [Magnusson 2009]
   * \return final step length
   */
  double
  computeStepLengthMT(
    const Eigen::Matrix<double, 6, 1> & x,
    Eigen::Matrix<double, 6, 1> & step_dir,
    double step_init,
    double step_max, double step_min,
    double & score,
    Eigen::Matrix<double, 6, 1> & score_gradient,
    Eigen::Matrix<double, 6, 6> & hessian,
    PointCloudSource & trans_cloud);

  /** \brief Auxiliary function used to determine endpoints of More-Thuente interval.
   * \note \f$ \psi(\alpha) \f$ in Equation 1.6 (Moore, Thuente 1994)
   * \param[in] a the step length, \f$ \alpha \f$ in More-Thuente (1994)
   * \param[in] f_a function value at step length a, \f$ \phi(\alpha) \f$ in More-Thuente (1994)
   * \param[in] f_0 initial function value, \f$ \phi(0) \f$ in Moore-Thuente (1994)
   * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
   * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
   * \return sufficient decrease value
   */
  inline double
  auxiliaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu = 1.e-4)
  {
    return f_a - f_0 - mu * g_0 * a;
  }

  /** \brief Auxiliary function derivative used to determine endpoints of More-Thuente interval.
   * \note \f$ \psi'(\alpha) \f$, derivative of Equation 1.6 (Moore, Thuente 1994)
   * \param[in] g_a function gradient at step length a, \f$ \phi'(\alpha) \f$ in More-Thuente (1994)
   * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
   * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
   * \return sufficient decrease derivative
   */
  inline double
  auxiliaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4)
  {
    return g_a - mu * g_0;
  }

  /** \brief The voxel grid generated from target cloud containing point means and covariances. */
  TargetGrid target_cells_;

  //double fitness_epsilon_;

  /** \brief The side length of voxels. */
  double resolution_;

  /** \brief The maximum step length. */
  double step_size_;

  /** \brief The ratio of outliers of points w.r.t. a normal distribution, Equation 6.7 [Magnusson 2009]. */
  double outlier_ratio_;

  /** \brief The normalization constants used fit the point distribution to a normal distribution, Equation 6.8 [Magnusson 2009]. */
  double gauss_d1_, gauss_d2_, gauss_d3_;

  /** \brief The probability score of the transform applied to the input cloud, Equation 6.9 and 6.10 [Magnusson 2009]. */
  double trans_probability_;

  /** \brief The first order derivative of the transformation of a point w.r.t. the transform vector, \f$ J_E \f$ in Equation 6.18 [Magnusson 2009]. */
  //      Eigen::Matrix<double, 3, 6> point_gradient_;

  /** \brief The second order derivative of the transformation of a point w.r.t. the transform vector, \f$ H_E \f$ in Equation 6.20 [Magnusson 2009]. */
  //      Eigen::Matrix<double, 18, 6> point_hessian_;

  int num_threads_;

  Eigen::Matrix<double, 6, 6> hessian_;
  std::vector<Eigen::Matrix4f> transformation_array_;

public:
  NeighborSearchMethod search_method;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif // PCL_REGISTRATION_NDT_H_
