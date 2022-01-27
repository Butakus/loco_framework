#include <iostream>
#include <iterator>

#include <geometry_msgs/msg/pose.hpp>
#include <loco_framework/pose.hpp>
#include <loco_framework/estimators/loco_estimator.hpp>

#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/math/ops_matrices.h>

using namespace loco;

std::random_device rd;
std::mt19937 random_generator(rd());
// std::mt19937 random_generator(42);

size_t population_size = 500;
size_t number_of_vehicles = 1;

// Ground truth
std::vector<PoseSE2> true_position(number_of_vehicles);
std::vector<PoseSE2> odometry(number_of_vehicles);


void print_poses(std::vector<PoseSE2> poses)
{
    for (const auto& p : poses)
    {
        std::cout << p << std::endl;
    }
}

void print_poses(std::vector<NoisyPoseSE2> poses, bool full=false)
{
    for (const auto& p : poses)
    {
        std::string s = full ? p.fullStr() : p.str();
        std::cout << s << std::endl;
    }
}


void init_poses()
{
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        double x = 4.0 * i;
        double y = 0.0;
        double a = deg_to_rad(0.0);
        true_position[i] = PoseSE2(x, y, a);
    }
}

void init_odometries()
{
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        double dx = 1.0 + 0.5 * i;
        double dy = 0.0;
        double da = deg_to_rad(0.0);
        odometry[i] = PoseSE2(dx, dy, da);
    }    
}

std::vector<NoisyPoseSE2> get_prior_estimations()
{
    std::vector<NoisyPoseSE2> prior_estimations(number_of_vehicles);
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        Eigen::Matrix<double, 3, 3> covariance;
        covariance << 1.0, 0, 0,
                      0, 1.0, 0,
                      0, 0, 0.005;
        NoisyPoseSE2 prior_estimation(true_position[i], covariance);
        // std::cout << "Prior " << i << ":\n" << prior_estimation.fullStr() << std::endl;
        prior_estimations[i] = prior_estimation.sample_mvn(random_generator);
    }
    return prior_estimations;
}

std::vector<NoisyPoseSE2> get_odometry()
{
    std::vector<NoisyPoseSE2> noisy_odometry(number_of_vehicles);
    for (size_t i = 0; i < number_of_vehicles; i++)
    {
        Eigen::Matrix<double, 3, 3> covariance;
        covariance << 0.001, 0, 0,
                      0, 0.0001, 0,
                      0, 0, 0.0005;
        NoisyPoseSE2 odom(odometry[i], covariance);
        noisy_odometry[i] = odom.sample_mvn(random_generator);

        // Update the position
        true_position[i] += odometry[i];
    }
    return noisy_odometry;
}

double drawGaussian1D_normalized()
{
    std::normal_distribution<> gaussian_distribution(0.0, 1.0);
    return gaussian_distribution(random_generator);
}


template <typename T, typename MATRIX>
void drawGaussianMultivariate(
    std::vector<T>& out_result, const MATRIX& cov)
{
    const size_t dim = cov.cols();
    if (cov.rows() != cov.cols())
        throw std::runtime_error(
            "drawGaussianMultivariate(): cov is not square.");
    MATRIX Z, D;
    std::vector<double> eigVals;
    cov.eig_symmetric(Z, eigVals, false /*sorted*/);
    // Set size of output vector:
    out_result.clear();
    out_result.resize(dim, 0);
    Eigen::Matrix<double, 3, 3> eigVals_eigen;
    eigVals_eigen << eigVals[0], 0, 0,
                     0, eigVals[1], 0,
                     0, 0, eigVals[2];
    D = mrpt::math::CMatrixDouble33(eigVals_eigen);
    /** Computes the eigenvalues/eigenvector decomposition of this matrix,
     *    so that: M = Z * D * Z<sup>T</sup>, where columns in Z are the
     *    eigenvectors and the diagonal matrix D contains the eigenvalues
     *    as diagonal elements, sorted in <i>ascending</i> order.
     */
    // cov.eigenVectors(Z, D);

    std::cout << "eigVecs (Z):\n" << Z << std::endl;
    std::cout << "eigVals (D):\n" << D << std::endl;
    // for (const auto& i : eigVals) std::cout << i << " ";
    // std::cout << std::endl;

    // Scale eigenvectors with eigenvalues:
    D = D.array().sqrt().matrix();
    Z.matProductOf_AB(Z, D);
    std::cout << "scaled eigVecs (Z):\n" << Z << std::endl;
    std::cout << "scaled eigVals (D):\n" << D << std::endl;
    for (size_t i = 0; i < dim; i++)
    {
        T rnd = drawGaussian1D_normalized();
        for (size_t d = 0; d < dim; d++)
            out_result[d] += (Z(d, i) * rnd);
    }
}

template <class VECTORLIKE, class COVMATRIX>
void drawGaussianMultivariate(
    VECTORLIKE& out_result, const COVMATRIX& cov)
{
    const size_t N = cov.rows();
    if (cov.rows() != cov.cols())
        throw std::runtime_error(
            "drawGaussianMultivariate(): cov is not square.");

    // Compute eigenvalues/eigenvectors of cov:
    COVMATRIX eigVecs;
    std::vector<typename COVMATRIX::Scalar> eigVals;
    cov.eig_symmetric(eigVecs, eigVals, false /*sorted*/);

    std::cout << "eigVecs:\n" << eigVecs << std::endl;
    std::cout << "eigVals:" << std::endl;
    for (const auto& i : eigVals) std::cout << i << " ";
    std::cout << std::endl;

    // Scale eigenvectors with eigenvalues:
    // D.Sqrt(); Z = Z * D; (for each column)
    for (typename COVMATRIX::Index c = 0; c < eigVecs.cols(); c++)
    {
        const auto s = std::sqrt(eigVals[c]);
        for (typename COVMATRIX::Index r = 0; r < eigVecs.rows(); r++)
        {
            std::cout << "c, r: " << c << ", " << r << std::endl;
            eigVecs(c, r) *= s;
            std::cout << "s: " << s << std::endl;
            std::cout << "eigVecs(c, r): " << eigVecs(c, r) << std::endl;
            std::cout << "eigVecs:\n" << eigVecs << std::endl;
        }
    }

    std::cout << "After scaling:" << std::endl;
    std::cout << "eigVecs:\n" << eigVecs << std::endl;
    std::cout << "eigVals:" << std::endl;
    for (const auto& i : eigVals) std::cout << i << " ";
    std::cout << std::endl;

    // Set size of output vector:
    out_result.assign(N, 0);

    for (size_t i = 0; i < N; i++)
    {
        typename COVMATRIX::Scalar rnd = drawGaussian1D_normalized();
        for (size_t d = 0; d < N; d++)
        {
            out_result[d] += eigVecs.coeff(d, i) * rnd;
        }
    }
}

void matrix_ref(double& x)
{
    x = x + 1;
}

int main()
{
    std::cout << "MRPT Pose Test" << std::endl;

    Eigen::Matrix<double, 3, 3> covariance_A;
    double A_x = 4.0;
    double A_y = 4.0;
    double A_angle = deg_to_rad(45);
    covariance_A << 0.1, 0.0, 0.0,
                  0.0, 0.1, 0.0,
                  0.0, 0.0, 0.005;
    double B_x = 3.0;
    double B_y = 4.0;
    double B_angle = deg_to_rad(0);
    Eigen::Matrix<double, 3, 3> covariance_B;
    covariance_B << 0.2, 0.0, 0.0,
                    0.0, 0.2, 0.0,
                    0.0, 0.0, 0.005;
    NoisyPoseSE2 loco_pose_A(PoseSE2(A_x, A_y, A_angle), covariance_A);
    std::cout << "Loco pose A:\n" << loco_pose_A.fullStr() << std::endl;
    NoisyPoseSE2 loco_pose_B(PoseSE2(B_x, B_y, B_angle), covariance_B);
    std::cout << "Loco pose B:\n" << loco_pose_B.fullStr() << std::endl;

    mrpt::poses::CPosePDFGaussian mrpt_pose_A(mrpt::poses::CPose2D(A_x, A_y, A_angle), mrpt::math::CMatrixDouble33(covariance_A));
    std::cout << "MRPT pose A:\n" << mrpt_pose_A << std::endl;
    mrpt::poses::CPosePDFGaussian mrpt_pose_B(mrpt::poses::CPose2D(B_x, B_y, B_angle), mrpt::math::CMatrixDouble33(covariance_B));
    std::cout << "MRPT pose B:\n" << mrpt_pose_B << std::endl;

    /*
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3> > solver =
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3> >(covariance_A);
    Eigen::Matrix<double, 3, 3> transform = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    // std::cout << "Loco eigen vectors:\n" << solver.eigenvectors() << std::endl;
    // std::cout << "Loco eigen values:\n" << solver.eigenvalues() << std::endl;
    Eigen::Matrix<double, 3, 3> diag = solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    // std::cout << "solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal():\n" << diag << std::endl;

    std::cout << "Loco pose MVN transform:" << std::endl;
    std::cout << loco_pose.mvn_sample_transform() << std::endl;

    std::cout << "Loco pose sample:" << std::endl;
    std::cout << loco_pose.sample_mvn(random_generator).fullStr() << std::endl;

    mrpt::math::CVectorDouble v;
    std::vector<double> vv;
    drawGaussianMultivariate(vv, mrpt_pose.cov);
    std::cout << "drawGaussianMultivariate:" << std::endl;
    for (const auto& i : vv) std::cout << i << " ";
    std::cout << std::endl;

    mrpt::poses::CPose2D mrpt_sample;
    mrpt_pose.drawSingleSample(mrpt_sample);
    std::cout << "MRPT noisy sample:" << std::endl;
    std::cout << mrpt_sample << std::endl;
    */
    

    std::cout << "##############################################" << std::endl;
    std::cout << "# Loco pose operations" << std::endl;
    std::cout << "##############################################" << std::endl;
    std::cout << "Loco A + B:\n" << (loco_pose_A + loco_pose_B).fullStr() << std::endl;
    std::cout << "Loco B - A:\n" << (loco_pose_B - loco_pose_A).fullStr() << std::endl;
    NoisyPoseSE2 loco_pose_A_inv = loco_pose_A.inverse();
    std::cout << "Loco A inverse:\n" << loco_pose_A_inv.fullStr() << std::endl;

    std::cout << "A + Ainv:\n" << (loco_pose_A + loco_pose_A_inv).fullStr() << std::endl;
    std::cout << "Ainv + B:\n" << (loco_pose_A_inv + loco_pose_B).fullStr() << std::endl;
    std::cout << "B + Ainv + A:\n" << (loco_pose_B + loco_pose_A_inv + loco_pose_A).fullStr() << std::endl;
    std::cout << "Ainv + A + B:\n" << (loco_pose_A_inv + loco_pose_A + loco_pose_B).fullStr() << std::endl;



    std::cout << "##############################################" << std::endl;
    std::cout << "# MRPT pose operations" << std::endl;
    std::cout << "##############################################" << std::endl;
    mrpt::poses::CPosePDFGaussian mrpt_pose_AB_comp = mrpt_pose_A + mrpt_pose_B;
    std::cout << "MRPT A + B:\n" << mrpt_pose_AB_comp << std::endl;
    /*

    mrpt::math::CMatrixDouble33 df_dA(mrpt::math::UNINITIALIZED_MATRIX), df_dB(mrpt::math::UNINITIALIZED_MATRIX);

    mrpt::poses::CPosePDF::jacobiansPoseComposition(
        mrpt_pose_A.mean,  // x
        mrpt_pose_B.mean,  // u
        df_dA, df_dB
    );

    std::cout << "df_dA:\n" << df_dA << std::endl;
    std::cout << "df_dA.T:\n" << df_dA.asEigen().transpose() << std::endl;
    std::cout << "df_dB:\n" << df_dB << std::endl;
    std::cout << "df_dB.T:\n" << df_dB.asEigen().transpose() << std::endl;

    auto new_cov = mrpt::math::multiply_HCHt(df_dA, mrpt_pose_A.cov) + mrpt::math::multiply_HCHt(df_dB, mrpt_pose_B.cov);
    std::cout << "df_dA * cov:\n" << df_dA.asEigen() * mrpt_pose_A.cov.asEigen() << std::endl;
    std::cout << "df_dA * cov * df_dA.T:\n" << df_dA.asEigen() * mrpt_pose_A.cov.asEigen() * df_dA.asEigen().transpose() << std::endl;
    std::cout << "new_cov A term:\n" << mrpt::math::multiply_HCHt(df_dA, mrpt_pose_A.cov) << std::endl;
    std::cout << "new_cov B term:\n" << mrpt::math::multiply_HCHt(df_dB, mrpt_pose_B.cov) << std::endl;
    std::cout << "new_cov:\n" << new_cov << std::endl;
    */

    // mrpt::poses::CPosePDFGaussian mrpt_pose_AB_sub = mrpt_pose_A - mrpt_pose_B;
    // std::cout << "A - B:\n" << mrpt_pose_AB_sub << std::endl;
    mrpt::poses::CPosePDFGaussian mrpt_pose_BA_sub = mrpt_pose_B - mrpt_pose_A;
    std::cout << "B - A:\n" << mrpt_pose_BA_sub << std::endl;

    mrpt::poses::CPosePDFGaussian mrpt_pose_A_inv;
    mrpt_pose_A.inverse(mrpt_pose_A_inv);
    std::cout << "Ainv:\n" << mrpt_pose_A_inv << std::endl;

    mrpt::poses::CPosePDFGaussian mrpt_pose_AAinv_comp = mrpt_pose_A + mrpt_pose_A_inv;
    std::cout << "A + Ainv:\n" << mrpt_pose_AAinv_comp << std::endl;

    mrpt::poses::CPosePDFGaussian mrpt_pose_AinvB_comp = mrpt_pose_A_inv + mrpt_pose_B;
    std::cout << "Ainv + B:\n" << mrpt_pose_AinvB_comp << std::endl;

    std::cout << "B + Ainv + A:\n" << mrpt_pose_B + mrpt_pose_A_inv + mrpt_pose_A << std::endl;
    std::cout << "Ainv + A + B:\n" << mrpt_pose_A_inv + mrpt_pose_A + mrpt_pose_B << std::endl;

    Eigen::Matrix<double, 3, 3> test_mat;
    test_mat << 1, 0, 5,
                0, 1, 3,
                0, 0, 1;
    std::cout << "test_mat:\n" << test_mat << std::endl;
    matrix_ref(test_mat(0, 2));
    std::cout << "test_mat:\n" << test_mat << std::endl;
    std::cout << "loco_pose_A:\n" << loco_pose_A << std::endl;
    matrix_ref(loco_pose_A.x());
    // loco_pose_A.x() = loco_pose_A.x() + 1;
    std::cout << "loco_pose_A:\n" << loco_pose_A << std::endl;

    return 0;
}
